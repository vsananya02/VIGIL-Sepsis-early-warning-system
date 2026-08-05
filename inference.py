# infrence
"""
VIGIL — inference.py
(24, 29) scaled array + static values  ->  risk, tier, flags.

Port of assess_patient() with every Colab dependency removed. Loads only
VIGIL_DEPLOY_v1.pt. Model classes below are VERBATIM from the reload script —
rename one layer and load_state_dict fails.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ MODEL DEFINITIONS
class GatedResidualNetwork(nn.Module):
    def __init__(s, i, h, o, dropout=0.1):
        super().__init__(); s.fc1 = nn.Linear(i, h); s.fc2 = nn.Linear(h, o)
        s.gate = nn.Linear(h, o); s.norm = nn.LayerNorm(o); s.drop = nn.Dropout(dropout)
        s.res = nn.Linear(i, o, bias=False) if i != o else nn.Identity()
    def forward(s, x):
        h = s.drop(F.elu(s.fc1(x)))
        return s.norm(torch.sigmoid(s.gate(h)) * s.fc2(h) + s.res(x))


class VariableSelectionNetwork(nn.Module):
    def __init__(s, n, d, dropout=0.1):
        super().__init__()
        s.grns = nn.ModuleList([GatedResidualNetwork(1, d, d, dropout) for _ in range(n)])
        s.wgrn = GatedResidualNetwork(n, d, n, dropout)
    def forward(s, x):
        if x.dim() == 3:
            B, S, V = x.shape
            vo = torch.stack([s.grns[i](x[:, :, i:i+1]) for i in range(V)], -1)
            w = F.softmax(s.wgrn(x.reshape(B*S, V)), -1).reshape(B, S, V)
            return (vo * w.unsqueeze(-2)).sum(-1), w
        B, V = x.shape
        vo = torch.stack([s.grns[i](x[:, i:i+1]) for i in range(V)], -1)
        w = F.softmax(s.wgrn(x), -1)
        return (vo * w.unsqueeze(1)).sum(-1), w


class SepsisTFT(nn.Module):
    def __init__(s, n_static, n_temporal, d_model=128, n_heads=4, dropout=0.2):
        super().__init__(); s.svsn = VariableSelectionNetwork(n_static, d_model, dropout)
        s.senc = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        s.ctx_h = nn.Linear(d_model, d_model); s.ctx_c = nn.Linear(d_model, d_model)
        s.tvsn = VariableSelectionNetwork(n_temporal, d_model, dropout)
        s.lstm = nn.LSTM(d_model, d_model, 2, batch_first=True, bidirectional=True, dropout=dropout)
        s.lnorm = nn.LayerNorm(d_model*2)
        s.lgrn = GatedResidualNetwork(d_model*2, d_model, d_model*2, dropout)
        s.attn = nn.MultiheadAttention(d_model*2, n_heads, dropout=dropout, batch_first=True)
        s.anorm = nn.LayerNorm(d_model*2)
        s.agrn = GatedResidualNetwork(d_model*2, d_model, d_model*2, dropout)
        s.ogrn = GatedResidualNetwork(d_model*2 + d_model, d_model, d_model, dropout)
        s.clf = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(s, temporal, static, return_attention=False):
        se, sw = s.svsn(static); sc = s.senc(se)
        h0 = torch.tanh(s.ctx_h(sc)).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.tanh(s.ctx_c(sc)).unsqueeze(0).repeat(2, 1, 1)
        hi = torch.cat([h0, torch.zeros_like(h0)], 0); ci = torch.cat([c0, torch.zeros_like(c0)], 0)
        te, tw = s.tvsn(temporal); lo, _ = s.lstm(te, (hi, ci)); lo = s.lgrn(s.lnorm(lo))
        ao, aw = s.attn(lo, lo, lo); ao = s.agrn(s.anorm(ao + lo))
        out = s.clf(s.ogrn(torch.cat([ao[:, -1, :], sc], -1))).squeeze(-1)
        return (out, tw, aw) if return_attention else out


class ReboundGRU(nn.Module):
    def __init__(s, n, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        s.proj = nn.Sequential(nn.Linear(n, 32), nn.LayerNorm(32), nn.GELU())
        s.gru = nn.GRU(32, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        s.norm = nn.LayerNorm(hidden*2)
        s.aW = nn.Linear(hidden*2, 64); s.av = nn.Linear(64, 1, bias=False)
        s.clf = nn.Sequential(nn.Linear(hidden*2, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(s, x):
        h = s.proj(x); g, _ = s.gru(h); g = s.norm(g)
        w = torch.softmax(s.av(torch.tanh(s.aW(g))), dim=1)
        return s.clf((w * g).sum(1)).squeeze(-1)


# ================================================================ LACTATE LAYER
class LactateTracker:
    """
    Per-patient, spans the WHOLE stay — not the 24h window. assess_patient()
    passes the full history, and clearance_pct compares against the first ever
    draw, so a rolling buffer would give different answers. Costs a few floats.
    """
    def __init__(self):
        self.draws = []                      # (hour, value) of DISTINCT readings

    def update(self, hour, lactate):
        if lactate is None or (isinstance(lactate, float) and np.isnan(lactate)):
            return
        if not self.draws or self.draws[-1][1] != lactate:   # undo forward-fill
            self.draws.append((float(hour), float(lactate)))

    def trajectory(self):
        if not self.draws:
            return {'status': 'NO_DATA', 'velocity': 0.0, 'clearance_pct': 0.0, 'n_measurements': 0}
        if len(self.draws) < 2:
            return {'status': 'ONE_READING', 'velocity': 0.0, 'clearance_pct': 0.0, 'n_measurements': 1}
        (h0, v0), (h1, v1) = self.draws[-2], self.draws[-1]
        dt = max(h1 - h0, 1.0)
        vel = (v1 - v0) / dt
        status = 'RISING' if vel > 0.02 else 'CLEARING' if vel < -0.02 else 'PLATEAU'
        first = self.draws[0][1]
        clr = (first - v1) / first * 100 if first > 0 else 0.0
        return {'status': status, 'velocity': round(float(vel), 3),
                'clearance_pct': round(float(clr), 1), 'n_measurements': len(self.draws)}


MORTALITY = {'CLEARING': 0.264, 'PLATEAU': 0.384, 'RISING': 0.765}
_odds = lambda p: p / (1 - p)
LOGODDS_SHIFT = {k: np.log(_odds(v) / _odds(MORTALITY['PLATEAU'])) for k, v in MORTALITY.items()}


def _adjust(base, status, vel):
    base = float(np.clip(base, 1e-4, 1 - 1e-4))
    z = np.log(base / (1 - base)) + LOGODDS_SHIFT.get(status, 0.0) * min(1.0, abs(vel) / 0.5)
    return float(1 / (1 + np.exp(-z)))


# ==================================================================== THE ENGINE
class VigilEngine:
    def __init__(self, bundle_path, device='cpu'):
        b = torch.load(bundle_path, map_location='cpu', weights_only=False)
        self.device = torch.device(device)

        hp = b['tft']['hparams']
        self.tft = SepsisTFT(hp['n_static'], hp['n_temporal'], hp['d_model'],
                             hp['n_heads'], hp['dropout']).to(self.device)
        self.tft.load_state_dict(b['tft']['state_dict']); self.tft.eval()

        self.rebound = ReboundGRU(b['rebound']['hparams']['n']).to(self.device)
        self.rebound.load_state_dict(b['rebound']['state_dict']); self.rebound.eval()

        self.fluid          = b['fluid']['model']
        self.FLUID_FEATURES = b['fluid']['features']
        self.calib          = b['calibrator']
        self.TEMPORAL       = b['TEMPORAL_FEATURES']
        self.REBOUND        = b['REBOUND_FEATURES']
        self.STATIC         = b['STATIC_FEATURES']
        self.tiers          = b['tiers']

        # column positions — every downstream model slices the same (24,29) array
        self._reb_idx   = [self.TEMPORAL.index(f) for f in self.REBOUND]
        self._fluid_idx = [self.TEMPORAL.index(f) for f in self.FLUID_FEATURES]

    def assess(self, temporal_29, static_values, lactate_traj):
        """
        temporal_29   : (24, 29) float32 from FeatureBuilder.build()
        static_values : dict keyed by STATIC_FEATURES (missing -> 0.0)
        lactate_traj  : dict from LactateTracker.trajectory()
        """
        Xt = torch.FloatTensor(temporal_29).unsqueeze(0).to(self.device)
        Xs = torch.FloatTensor([[float(static_values.get(f, 0.0)) for f in self.STATIC]]).to(self.device)

        with torch.no_grad():
            raw = float(torch.sigmoid(self.tft(Xt, Xs)).cpu())
            sepsis_risk = float(self.calib.predict([raw])[0])
            reb_in = torch.FloatTensor(temporal_29[:, self._reb_idx]).unsqueeze(0).to(self.device)
            rebound_risk = float(torch.sigmoid(self.rebound(reb_in)).cpu())

        fluid_row = temporal_29[-1:, self._fluid_idx]
        fluid_responsive = float(self.fluid.predict_proba(fluid_row)[0][1])

        adjusted = _adjust(sepsis_risk, lactate_traj['status'], lactate_traj['velocity'])

        flags = []
        if lactate_traj['status'] == 'RISING': flags.append('lactate rising')
        if rebound_risk > 0.5:                flags.append('rebound risk')
        if adjusted > sepsis_risk + 0.05:     flags.append('lactate-amplified')

        if   adjusted >= self.tiers['RED']:    tier = 'RED'
        elif adjusted >= self.tiers['ORANGE']: tier = 'ORANGE'
        elif adjusted >= self.tiers['YELLOW']: tier = 'YELLOW'
        else:                                  tier = 'GREEN'

        return {'tier': tier,
                'sepsis_risk': round(sepsis_risk, 3),
                'sepsis_risk_uncalibrated': round(raw, 3),
                'lactate_adjusted_risk': round(adjusted, 3),
                'lactate_status': lactate_traj['status'],
                'lactate_velocity': lactate_traj['velocity'],
                'lactate_clearance_pct': lactate_traj['clearance_pct'],
                'rebound_risk': round(rebound_risk, 3),
                'fluid_responsive': round(fluid_responsive, 3),
                'flags': flags}


# ==============================================================================
# ACCEPTANCE TEST — must reproduce your verification run exactly.
#   38066951 -> 98% RED, rebound 72%, fluid 33%
#   31741253 ->  0% GREEN
# ==============================================================================
def acceptance_test(mimic_scaled, static_lookup, test_ids, bundle_path, n=8):
    eng = VigilEngine(bundle_path)
    print(f"{'stay_id':>10} {'tier':>7} {'sepsis':>7} {'rebound':>8} {'fluid':>7}  flags")
    for sid in list(test_ids)[:n]:
        p = mimic_scaled[mimic_scaled['stay_id'] == int(sid)].sort_values('hours_in_icu')
        if len(p) < 6:
            continue
        M = p[eng.TEMPORAL].values.astype(np.float32)[-24:]
        if len(M) < 24:
            M = np.vstack([np.zeros((24 - len(M), M.shape[1]), np.float32), M])

        lt = LactateTracker()
        for h, v in zip(p['hours_in_icu'].values, p['lactate_raw'].values):
            lt.update(h, v)

        r = eng.assess(M, static_lookup.get(int(sid), {}), lt.trajectory())
        print(f"{sid:>10} {r['tier']:>7} {r['sepsis_risk']*100:>6.0f}% "
              f"{r['rebound_risk']*100:>7.0f}% {r['fluid_responsive']*100:>6.0f}%  "
              f"{', '.join(r['flags']) or 'none'}")
    print("\nCompare against your Step-A verification run. Every row must match.")


# In Colab:
#   acceptance_test(mimic_scaled, static_lookup, test_ids,
#                   os.path.join(SAVE_DIR, 'VIGIL_DEPLOY_v1.pt'))
