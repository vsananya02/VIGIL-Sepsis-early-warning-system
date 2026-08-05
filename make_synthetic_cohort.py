"""
VIGIL - make_synthetic_cohort.py          (SyntheticSource Lite)

Writes synthetic_cohort.json with the SAME schema as replay_cohort.json,
so vitals_source.py loads it with zero code changes.

WHY THIS EXISTS
    PhysioNet DUA forbids serving real MIMIC data publicly.
    Local  = replay_cohort.json   (real patients, real trajectories)
    Public = synthetic_cohort.json (invented patients, safe to deploy)

DESIGN RULE - these trajectories are written from PHYSIOLOGY, never tuned
against model output. Whatever VIGIL scores them, that score is reported
as-is. A synthetic cohort tuned until the model looks good proves nothing.

Run:  python make_synthetic_cohort.py
"""
import json
import os

import numpy as np

SEED = 7
OUT = "synthetic_cohort.json"
BUNDLE = "VIGIL_DEPLOY_v1.pt"          # optional - only for the STATIC key list

RAW_KEYS = ["hr", "rr", "sbp", "dbp", "o2_sat", "lactate", "wbc", "creatinine", "temp"]

rng = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════
def ramp(n, start, end, begin, finish):
    """Flat at `start` until hour `begin`, linear to `end` by `finish`, flat after."""
    out = np.full(n, float(start))
    for t in range(n):
        if t <= begin:
            out[t] = start
        elif t >= finish:
            out[t] = end
        else:
            frac = (t - begin) / max(finish - begin, 1)
            out[t] = start + (end - start) * frac
    return out


def jitter(series, sd_series):
    """Add per-hour gaussian noise. sd may vary over time (HRV collapses in sepsis)."""
    return series + rng.normal(0.0, 1.0, len(series)) * sd_series


def lab_schedule(series, every, n):
    """
    Labs are not drawn hourly. Sample the underlying curve every `every` hours
    and forward-fill between draws - exactly how MIMIC/eICU behave.
    """
    out = np.empty(n)
    last = series[0]
    for t in range(n):
        if t % every == 0:
            last = series[t]
        out[t] = last
    return out


def build(display_id, archetype, n_hours, start_hod, curves, hr_sd, onset_hour=None):
    """Assemble one patient dict in replay_cohort.json format."""
    rows = []
    for t in range(n_hours):
        row = {
            "hours_in_icu": t,
            "hour": int((start_hod + t) % 24),
        }
        for k in RAW_KEYS:
            row[k] = round(float(curves[k][t]), 2)
        rows.append(row)

    return {
        "display_id": display_id,
        "archetype": archetype,
        "n_hours": n_hours,
        "source": "SYNTHETIC - physiology-driven, no patient data",
        "static": STATIC_VEC,
        "ground_truth": {"onset_hour": onset_hour, "peak_risk": None},
        "rows": rows,
    }


# ══════════════════════════════════════════════════════════════════════
# static feature vector (all zeros except the three we can define)
# ══════════════════════════════════════════════════════════════════════
STATIC_KEYS = ["age_normalized", "gender_encoded", "baseline_creatinine_norm"]
if os.path.exists(BUNDLE):
    try:
        import torch
        _b = torch.load(BUNDLE, map_location="cpu", weights_only=False)
        STATIC_KEYS = _b["STATIC_FEATURES"]
        print(f"[ok] read {len(STATIC_KEYS)} static keys from {BUNDLE}")
    except Exception as e:
        print(f"[warn] could not read bundle ({e}); using 3 default static keys")

# 71-year-old male, baseline creatinine 1.0
STATIC_VEC = {k: 0.0 for k in STATIC_KEYS}
STATIC_VEC["age_normalized"] = (71 - 65) / 15
STATIC_VEC["gender_encoded"] = 1.0
STATIC_VEC["baseline_creatinine_norm"] = (1.0 - 1.0) / 1.5


# ══════════════════════════════════════════════════════════════════════
# SYN-01  STABLE  - nothing happens. Tests: no false alarms.
# ══════════════════════════════════════════════════════════════════════
n = 72
sd = np.full(n, 5.0)                       # healthy HRV all the way through
c = {
    "hr":         jitter(np.full(n, 76.0), sd),
    "rr":         jitter(np.full(n, 16.0), np.full(n, 1.5)),
    "sbp":        jitter(np.full(n, 122.0), np.full(n, 6.0)),
    "dbp":        jitter(np.full(n, 74.0), np.full(n, 4.0)),
    "o2_sat":     np.clip(jitter(np.full(n, 97.0), np.full(n, 0.9)), 88, 100),
    "temp":       jitter(np.full(n, 36.9), np.full(n, 0.2)),
    "lactate":    lab_schedule(jitter(np.full(n, 1.1), np.full(n, 0.15)), 12, n),
    "wbc":        lab_schedule(jitter(np.full(n, 7.2), np.full(n, 0.5)), 24, n),
    "creatinine": lab_schedule(jitter(np.full(n, 0.9), np.full(n, 0.05)), 24, n),
}
p_stable = build("SYN-01", "STABLE", n, 8, c, sd)


# ══════════════════════════════════════════════════════════════════════
# SYN-02  DETERIORATING - textbook sepsis. Everything drifts wrong.
#         HRV collapses as autonomic tone fails (sd 5.0 -> 1.5).
# ══════════════════════════════════════════════════════════════════════
n = 80
sd = ramp(n, 5.0, 1.5, 38, 60)             # low_hrv_flag should fire late
c = {
    "hr":         jitter(ramp(n, 78, 112, 38, 62), sd),
    "rr":         jitter(ramp(n, 16, 25, 40, 64), np.full(n, 1.4)),
    "sbp":        jitter(ramp(n, 121, 96, 40, 64), np.full(n, 6.0)),
    "dbp":        jitter(ramp(n, 74, 56, 40, 64), np.full(n, 4.0)),
    "o2_sat":     np.clip(jitter(ramp(n, 97, 92, 44, 68), np.full(n, 0.9)), 85, 100),
    "temp":       jitter(ramp(n, 36.9, 38.7, 40, 60), np.full(n, 0.25)),
    "lactate":    lab_schedule(ramp(n, 1.2, 3.6, 42, 66), 12, n),
    "wbc":        lab_schedule(ramp(n, 8.0, 16.5, 40, 64), 24, n),
    "creatinine": lab_schedule(ramp(n, 1.0, 1.8, 42, 66), 24, n),
}
p_deteriorating = build("SYN-02", "DETERIORATING", n, 14, c, sd, onset_hour=58)


# ══════════════════════════════════════════════════════════════════════
# SYN-03  CRYPTIC  - THE DEMO CASE. Modelled on ICU-09.
#         HR, temp, WBC, lactate all stay NORMAL the whole stay.
#         Only creatinine climbs and MAP quietly erodes.
#         SIRS scores 0 here. This is the patient the thesis is about.
# ══════════════════════════════════════════════════════════════════════
n = 80
sd = ramp(n, 5.0, 2.4, 34, 58)
# HARD CLIPS below every SIRS threshold. Jitter must never accidentally make
# this patient SIRS-positive - if it does, the demo case stops being cryptic.
c = {
    "hr":         np.clip(jitter(ramp(n, 76, 84, 36, 60), sd), 58, 88.5),
    "rr":         np.clip(jitter(ramp(n, 15, 17.5, 38, 62), np.full(n, 1.0)), 11, 19.4),
    "sbp":        jitter(ramp(n, 124, 103, 34, 58), np.full(n, 5.5)),
    "dbp":        jitter(ramp(n, 76, 61, 34, 58), np.full(n, 3.5)),
    "o2_sat":     np.clip(jitter(np.full(n, 96.0), np.full(n, 0.9)), 88, 100),
    "temp":       np.clip(jitter(np.full(n, 37.0), np.full(n, 0.18)), 36.3, 37.7),
    "lactate":    lab_schedule(jitter(np.full(n, 1.1), np.full(n, 0.12)), 12, n),
    "wbc":        lab_schedule(np.clip(jitter(np.full(n, 7.0), np.full(n, 0.4)), 4.6, 11.4), 24, n),
    "creatinine": lab_schedule(ramp(n, 0.9, 1.75, 34, 60), 24, n),   # the ONLY signal
}
p_cryptic = build("SYN-03", "CRYPTIC", n, 3, c, sd, onset_hour=56)


# ══════════════════════════════════════════════════════════════════════
# SYN-04  NONSEPSIS_MIMIC - looks septic early, then resolves.
#         Fever + tachycardia + high WBC, but organs stay fine.
#         Tests specificity. An honest demo includes one of these.
# ══════════════════════════════════════════════════════════════════════
n = 64
sd = np.full(n, 4.5)
spike = np.zeros(n)
spike[30:44] = 1.0
for t in range(44, 54):
    spike[t] = (54 - t) / 10.0
c = {
    "hr":         jitter(78 + 28 * spike, sd),
    "rr":         jitter(16 + 6 * spike, np.full(n, 1.5)),
    "sbp":        jitter(120 - 8 * spike, np.full(n, 6.0)),
    "dbp":        jitter(73 - 5 * spike, np.full(n, 4.0)),
    "o2_sat":     np.clip(jitter(97 - 2 * spike, np.full(n, 0.9)), 88, 100),
    "temp":       jitter(36.9 + 1.7 * spike, np.full(n, 0.25)),
    "lactate":    lab_schedule(jitter(np.full(n, 1.2), np.full(n, 0.15)), 12, n),
    "wbc":        lab_schedule(8.0 + 6.0 * spike, 24, n),
    "creatinine": lab_schedule(jitter(np.full(n, 0.9), np.full(n, 0.05)), 24, n),
}
p_nonsepsis = build("SYN-04", "NONSEPSIS_MIMIC", n, 20, c, sd)


# ══════════════════════════════════════════════════════════════════════
# write + self-check
# ══════════════════════════════════════════════════════════════════════
patients = [p_stable, p_deteriorating, p_cryptic, p_nonsepsis]
cohort = {
    "meta": {
        "source": "SYNTHETIC - generated from physiology, contains NO patient data",
        "seed": SEED,
        "n_patients": len(patients),
        "selection": "trajectories written from clinical priors; model measured after, never tuned against",
        "restriction": "none - safe to deploy publicly",
    },
    "patients": patients,
}

with open(OUT, "w") as f:
    json.dump(cohort, f)

print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e3:.0f} KB, {len(patients)} patients)\n")


def rolling_std(x, w=4):
    """Mirror features.py: rolling(4, min_periods=2).std()"""
    out = np.full(len(x), np.nan)
    for t in range(1, len(x)):
        seg = x[max(0, t - w + 1): t + 1]
        if len(seg) >= 2:
            out[t] = np.std(seg, ddof=1)
    return out


print("SELF-CHECK")
print(f"  {'id':<8}{'archetype':<18}{'hrs':>5}{'HRV med':>9}{'HRV end':>9}{'SIRS max':>10}")
print("  " + "-" * 60)
ok = True
for p in patients:
    hr = np.array([r["hr"] for r in p["rows"]])
    rr = np.array([r["rr"] for r in p["rows"]])
    tp = np.array([r["temp"] for r in p["rows"]])
    wb = np.array([r["wbc"] for r in p["rows"]])

    hrv = rolling_std(hr)
    hrv_med = np.nanmedian(hrv)
    hrv_end = np.nanmedian(hrv[-12:])

    sirs = (((tp > 38.0) | (tp < 36.0)).astype(int) + (hr > 90).astype(int)
            + (rr > 20).astype(int) + ((wb > 12) | (wb < 4)).astype(int))

    print(f"  {p['display_id']:<8}{p['archetype']:<18}{p['n_hours']:>5}"
          f"{hrv_med:>9.2f}{hrv_end:>9.2f}{int(sirs.max()):>10}")

    for r in p["rows"]:
        for k in RAW_KEYS:
            if not np.isfinite(r[k]):
                print(f"  FAIL - non-finite {k} in {p['display_id']}")
                ok = False

    if p["archetype"] == "STABLE" and hrv_med < 3.0:
        print(f"  FAIL - {p['display_id']} HRV {hrv_med:.2f} < 3.0; low_hrv_flag would fire on a healthy patient")
        ok = False

    if p["archetype"] == "CRYPTIC" and sirs.max() >= 2:
        print(f"  FAIL - {p['display_id']} is SIRS-positive; it is not cryptic")
        ok = False

crypt = [p for p in patients if p["archetype"] == "CRYPTIC"][0]
crows = crypt["rows"]
print(f"\n  CRYPTIC check (SYN-03) - must look normal on every SIRS criterion:")
print(f"    max HR      {max(r['hr'] for r in crows):6.1f}   (SIRS fires >90)")
print(f"    max RR      {max(r['rr'] for r in crows):6.1f}   (SIRS fires >20)")
print(f"    max temp    {max(r['temp'] for r in crows):6.1f}   (SIRS fires >38.0)")
print(f"    max WBC     {max(r['wbc'] for r in crows):6.1f}   (SIRS fires >12)")
print(f"    max lactate {max(r['lactate'] for r in crows):6.1f}   (normal <2.0)")
print(f"    creatinine  {crows[0]['creatinine']:.2f} -> {crows[-1]['creatinine']:.2f}   <- the only abnormal trend")

print(f"\n  {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED - fix before deploying'}")
print("\n  Next: run this cohort through features.py + inference.py and report")
print("  whatever risk scores come out. Do NOT edit these numbers to improve them.")
