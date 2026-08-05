"""
VIGIL — vitals_source.py

The clock. Hands out ONE hour per patient per tick, keeps a rolling buffer,
and runs the full model stack on it.

Needs only: VIGIL_DEPLOY_v1.pt, replay_cohort.json, features.py, inference.py.
No mimic_scaled. No reload script.
"""
import json
from collections import deque

import numpy as np

# from features import FeatureBuilder, BUFFER
# from inference import VigilEngine, LactateTracker

BUFFER_HOURS = 32          # 24 for the model + 8 lookback for window features
from features import FeatureBuilder, BUFFER
from inference import VigilEngine, LactateTracker


# ============================================================ THE SOURCE
class MimicReplaySource:
    """Reads the cohort file and releases one hour at a time."""

    def __init__(self, cohort_path):
        with open(cohort_path) as f:
            data = json.load(f)
        self.patients = {p['display_id']: p for p in data['patients']}
        self.cursor = {pid: 0 for pid in self.patients}      # which hour is next

    def ids(self):
        return list(self.patients)

    def static(self, pid):
        return self.patients[pid]['static']

    def total_hours(self, pid):
        return self.patients[pid]['n_hours']

    def next_row(self, pid):
        """Returns the next hour for this patient, or None if their stay ended."""
        i = self.cursor[pid]
        rows = self.patients[pid]['rows']
        if i >= len(rows):
            return None
        self.cursor[pid] += 1
        return rows[i]

    def reset(self, pid=None):
        for k in ([pid] if pid else self.cursor):
            self.cursor[k] = 0


# =========================================================== THE BUFFER
class PatientState:
    """One patient's rolling window plus their whole-stay lactate log."""

    def __init__(self, pid, static):
        self.pid = pid
        self.static = static
        self.buffer = deque(maxlen=BUFFER_HOURS)
        self.lactate = LactateTracker()
        self.hours_seen = 0
        self.history = []            # risk score per hour, for the sparkline
        self.active = True

    def add(self, row):
        self.buffer.append(row)
        self.hours_seen += 1
        self.lactate.update(row['hours_in_icu'], row['lactate'])

    @property
    def warming_up(self):
        return self.hours_seen < 24


# =========================================================== THE MONITOR
class VigilMonitor:
    """Ties source + buffers + models together. One tick = one clinical hour."""

    def __init__(self, bundle_path, source):
        self.fb = FeatureBuilder(bundle_path)
        self.engine = VigilEngine(bundle_path)
        self.source = source
        self.states = {pid: PatientState(pid, source.static(pid))
                       for pid in source.ids()}
        self.clock = 0

    def tick(self):
        """Advance every patient by one hour. Returns a list of results."""
        self.clock += 1
        out = []

        for pid, st in self.states.items():
            if not st.active:
                continue
            row = self.source.next_row(pid)
            if row is None:                       # stay finished
                st.active = False
                continue
            st.add(row)

            if st.warming_up:
                out.append({'patient_id': pid, 'status': 'WARMING_UP',
                            'hours_seen': st.hours_seen, 'needed': 24,
                            'latest_vitals': row})
                st.history.append(None)
                continue

            M = self.fb.build(list(st.buffer))
            r = self.engine.assess(M, st.static, st.lactate.trajectory())
            st.history.append(r['lactate_adjusted_risk'])

            r.update({'patient_id': pid, 'status': 'MONITORING',
                      'hours_seen': st.hours_seen, 'latest_vitals': row,
                      'risk_history': st.history[-48:]})
            out.append(r)

        return out

    def any_active(self):
        return any(s.active for s in self.states.values())


# ==============================================================================
# ACCEPTANCE TEST
# Replay one patient hour by hour. Peak risk must match the ground_truth
# recorded in the cohort file. If it does, the streaming pipeline is correct.
# ==============================================================================
def replay_one(bundle_path, cohort_path, display_id=None):
    src = MimicReplaySource(cohort_path)
    with open(cohort_path) as f:
        cohort = json.load(f)

    if display_id is None:                        # default: a CRYPTIC patient
        display_id = next((p['display_id'] for p in cohort['patients']
                           if p['archetype'] == 'CRYPTIC'), src.ids()[0])

    truth = next(p for p in cohort['patients'] if p['display_id'] == display_id)
    mon = VigilMonitor(bundle_path, src)
    for pid in list(mon.states):                  # isolate this one patient
        if pid != display_id:
            mon.states[pid].active = False

    peak, fired_at = 0.0, None
    while mon.any_active():
        for r in mon.tick():
            if r['status'] != 'MONITORING':
                continue
            risk = r['lactate_adjusted_risk']
            if risk > peak:
                peak = risk
            if fired_at is None and risk >= 0.75:
                fired_at = r['latest_vitals']['hours_in_icu']

    exp = truth['ground_truth']['peak_risk']
    onset = truth['ground_truth']['onset_hour']
    lead = (onset - fired_at) if (onset is not None and fired_at is not None) else None

    print(f"  patient      : {display_id}  ({truth['archetype']}, {truth['n_hours']}h)")
    print(f"  peak risk    : {peak*100:.0f}%   expected {exp*100:.0f}%")
    print(f"  first RED    : {('h'+str(fired_at)) if fired_at is not None else 'never'}")
    print(f"  true onset   : {('h'+str(onset)) if onset is not None else 'none'}")
    print(f"  lead time    : {(str(lead)+'h') if lead is not None else '-'}")
    print(f"\n  {'PASS — streaming matches offline' if abs(peak-exp) < 0.02 else 'MISMATCH — check buffer order'}")
    return peak


# In Colab:
#   replay_one(os.path.join(SAVE_DIR,'VIGIL_DEPLOY_v1.pt'),
#              os.path.join(SAVE_DIR,'replay_cohort.json'))
