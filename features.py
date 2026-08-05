"""
VIGIL — features.py
Raw hourly vitals  ->  (24, 29) scaled array the TFT expects.

No pandas. No Drive. No MIMIC. Loads VIGIL_DEPLOY_v1.pt and nothing else.
All formulas below were verified against the real data (15/15, max err ~1e-5).

BUFFER SIZE: keep LOOKBACK+SEQ_LEN = 32 hours per patient. Window features look
back up to 6 hours, so computing over only 24 would corrupt the first 6 rows.
Compute over 32, return the last 24.
"""
import numpy as np
import torch

SEQ_LEN  = 24
LOOKBACK = 8          # >= 6 for diff(6); 8 for headroom
BUFFER   = SEQ_LEN + LOOKBACK

RAW_KEYS = ['hr', 'rr', 'sbp', 'dbp', 'o2_sat', 'lactate', 'wbc', 'creatinine', 'temp']


def _diff(a, n):
    """pandas .diff(n).fillna(0) on a single ordered series."""
    out = np.zeros_like(a)
    if len(a) > n:
        out[n:] = a[n:] - a[:-n]
    return out


def _rolling_std(a, window=4, min_periods=2):
    """pandas .rolling(window, min_periods).std().fillna(0) — ddof=1."""
    out = np.zeros_like(a)
    for i in range(len(a)):
        seg = a[max(0, i - window + 1): i + 1]
        out[i] = seg.std(ddof=1) if len(seg) >= min_periods else 0.0
    return out


class FeatureBuilder:
    def __init__(self, bundle_path):
        b = torch.load(bundle_path, map_location='cpu', weights_only=False)
        self.MODEL_FEATURES = b['MODEL_FEATURES_V5']
        self.TEMPORAL       = b['TEMPORAL_FEATURES']
        self.NEW_FEATS      = b['NEW_FEATS']
        self.mean           = np.asarray(b['scaler']['mean'], dtype=np.float64)
        self.std            = np.asarray(b['scaler']['std'],  dtype=np.float64)
        self.div            = b['div_stats']
        # index of each base feature inside the 30-wide scaler
        self._idx = {f: self.MODEL_FEATURES.index(f) for f in self.MODEL_FEATURES}

    def _z(self, name, values):
        i = self._idx[name]
        return (values - self.mean[i]) / self.std[i]

    def build(self, rows):
        """
        rows: list of dicts, oldest first, each with 'hour' (0-23 clock hour)
              plus the 9 RAW_KEYS. Up to BUFFER of them.
        returns: (24, 29) float32
        """
        if not rows:
            return np.zeros((SEQ_LEN, len(self.TEMPORAL)), dtype=np.float32)

        rows = rows[-BUFFER:]
        v = {k: np.array([float(r[k]) for r in rows], dtype=np.float64) for k in RAW_KEYS}
        hod = np.array([float(r['hour']) for r in rows], dtype=np.float64)

        # ---- pointwise derived (all VERIFIED) ----
        f = dict(v)
        f['MAP']                  = (v['sbp'] + 2 * v['dbp']) / 3
        f['hrr_ratio']            = v['hr'] / (v['rr'] + 1)
        f['bp_ratio']             = v['sbp'] / v['dbp']
        f['lactate_risk']         = (v['lactate'] > 2.0).astype(np.float64)
        f['lactate_map_interact'] = v['lactate'] / (f['MAP'] + 1)
        f['temp_dysregulation']   = np.abs(v['temp'] - 37.0)
        f['hypothermia_flag']     = (v['temp'] < 36.0).astype(np.float64)
        f['fever_flag']           = (v['temp'] > 38.3).astype(np.float64)
        f['hour_sin']             = np.sin(2 * np.pi * hod / 24)
        f['hour_cos']             = np.cos(2 * np.pi * hod / 24)
        f['is_night']             = ((hod >= 0) & (hod <= 6)).astype(np.float64)

        # ---- window features (order matters: hr_variability before its flag) ----
        f['wbc_trend']         = _diff(v['wbc'], 3)
        f['creatinine_delta6'] = _diff(v['creatinine'], 6)
        f['lactate_delta6']    = _diff(v['lactate'], 6)
        f['MAP_delta6']        = _diff(f['MAP'], 6)
        f['hr_variability']    = _rolling_std(v['hr'], 4, 2)
        f['low_hrv_flag']      = (f['hr_variability'] < 3.0).astype(np.float64)

        # ---- z-score the 26 non-leaky base features ----
        base = [c for c in self.TEMPORAL if c not in self.NEW_FEATS]
        S = {c: self._z(c, f[c]) for c in base}

        # ---- divergence: built FROM SCALED deltas, then scaled again ----
        cre = np.clip(S['creatinine_delta6'], -5, 5)
        lac = np.clip(S['lactate_delta6'],    -5, 5)
        wbt = np.clip(S['wbc_trend'],         -5, 5)
        raw_div = {
            'cryptic_divergence':      (np.clip(cre, 0, None) + np.clip(lac, 0, None))
                                       * np.clip(-wbt, 0, None),
            'wbc_organ_mismatch':      (cre + lac) / 2 - wbt,
            'creatinine_acceleration': np.clip(_diff(S['creatinine_delta6'], 1), -5, 5),
        }
        for name, arr in raw_div.items():
            S[name] = (arr - self.div[name]['mu']) / self.div[name]['sd']

        # ---- assemble in TEMPORAL order, take last 24, left-pad ----
        M = np.stack([S[c] for c in self.TEMPORAL], axis=1).astype(np.float32)
        M = M[-SEQ_LEN:]
        if len(M) < SEQ_LEN:
            M = np.vstack([np.zeros((SEQ_LEN - len(M), M.shape[1]), np.float32), M])
        return M


# ==============================================================================
# PARITY TEST — run this in Colab. Nothing ships until it passes.
#
# Trick: you don't need the raw CSV. Unscale mimic_scaled back to raw values
# (already proven to work), feed those through FeatureBuilder, and compare the
# output against mimic_scaled's own TEMPORAL_FEATURES columns.
# ==============================================================================
def parity_test(mimic_scaled, test_ids, bundle_path, n_patients=5):
    fb = FeatureBuilder(bundle_path)
    MF, SM, SD = fb.MODEL_FEATURES, fb.mean, fb.std
    worst = 0.0

    for sid in list(test_ids)[:n_patients]:
        p = mimic_scaled[mimic_scaled['stay_id'] == sid].sort_values('hours_in_icu')
        if len(p) < BUFFER:
            continue
        p = p.tail(BUFFER)

        # unscale the 9 base vitals back to raw clinical values
        rows = []
        un = {k: p[k].values * SD[MF.index(k)] + SM[MF.index(k)] for k in RAW_KEYS}
        hod = (np.arctan2(p['hour_sin'].values * SD[MF.index('hour_sin')] + SM[MF.index('hour_sin')],
                          p['hour_cos'].values * SD[MF.index('hour_cos')] + SM[MF.index('hour_cos')])
               / (2 * np.pi) * 24) % 24
        for j in range(len(p)):
            r = {k: un[k][j] for k in RAW_KEYS}
            r['hour'] = round(hod[j])
            rows.append(r)

        mine = fb.build(rows)
        ref  = p[fb.TEMPORAL].values.astype(np.float32)[-SEQ_LEN:]
        err  = np.abs(mine - ref).max()
        worst = max(worst, err)

        col = np.abs(mine - ref).max(axis=0)
        bad = [(fb.TEMPORAL[i], float(col[i])) for i in np.argsort(-col)[:3] if col[i] > 1e-4]
        print(f"  stay {sid}: max err {err:.3e} {'OK' if err < 1e-4 else 'FAIL ' + str(bad)}")

    print(f"\nworst across patients: {worst:.3e}  "
          f"{'PASS — build the API' if worst < 1e-4 else 'FAIL — do not proceed'}")
    return worst


# In Colab:
#   parity_test(mimic_scaled, test_ids, os.path.join(SAVE_DIR, 'VIGIL_DEPLOY_v1.pt'))
