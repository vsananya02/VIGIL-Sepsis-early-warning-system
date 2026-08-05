"""
VIGIL — check_cohort.py
Reads replay_cohort.json and prints it in a readable form.

Usage (in the VIGIL folder):
    python check_cohort.py            -> table of all 15 patients
    python check_cohort.py ICU-12     -> one patient in detail
"""
import json, sys

with open('replay_cohort.json') as f:
    cohort = json.load(f)

pts = cohort['patients']

# ---------------------------------------------------------- one patient
if len(sys.argv) > 1:
    want = sys.argv[1].upper()
    p = next((x for x in pts if x['display_id'] == want), None)
    if p is None:
        print(f"No patient called {want}. Available: "
              + ', '.join(x['display_id'] for x in pts))
        sys.exit()

    gt = p['ground_truth']
    print(f"\n  {p['display_id']}   ({p['archetype']})")
    print(f"  source stay_id   : {p['source_stay_id']}")
    print(f"  hours in ICU     : {p['n_hours']}")
    print(f"  true onset hour  : {gt['onset_hour'] if gt['onset_hour'] is not None else 'never'}")
    print(f"  peak risk (offline): {gt['peak_risk']*100:.0f}%   <- dashboard should match this")

    print(f"\n  first 3 hours of raw vitals (exactly what the dashboard shows):")
    print(f"  {'h':>4} {'hr':>5} {'rr':>5} {'sbp':>5} {'dbp':>5} {'spo2':>5} "
          f"{'lact':>6} {'wbc':>6} {'creat':>6} {'temp':>6}")
    for r in p['rows'][:3]:
        print(f"  {r['hours_in_icu']:>4} {r['hr']:>5} {r['rr']:>5} {r['sbp']:>5} "
              f"{r['dbp']:>5} {r['o2_sat']:>5} {r['lactate']:>6} {r['wbc']:>6} "
              f"{r['creatinine']:>6} {r['temp']:>6}")
    print(f"\n  last 3 hours:")
    for r in p['rows'][-3:]:
        print(f"  {r['hours_in_icu']:>4} {r['hr']:>5} {r['rr']:>5} {r['sbp']:>5} "
              f"{r['dbp']:>5} {r['o2_sat']:>5} {r['lactate']:>6} {r['wbc']:>6} "
              f"{r['creatinine']:>6} {r['temp']:>6}")
    sys.exit()

# ---------------------------------------------------------- all patients
print(f"\n  {cohort['meta']['n_patients']} patients from {cohort['meta']['source']}\n")
print(f"  {'id':<8}{'archetype':<18}{'hours':>6}{'onset':>8}{'peak risk':>11}")
print("  " + "-" * 51)
for p in sorted(pts, key=lambda x: -x['ground_truth']['peak_risk']):
    gt = p['ground_truth']
    onset = f"h{gt['onset_hour']}" if gt['onset_hour'] is not None else "-"
    print(f"  {p['display_id']:<8}{p['archetype']:<18}{p['n_hours']:>6}"
          f"{onset:>8}{gt['peak_risk']*100:>10.0f}%")

print(f"\n  'peak risk' is what the model scored offline.")
print(f"  The dashboard should reach the same number for each patient.")
print(f"\n  For one patient in detail:  python check_cohort.py ICU-12")
