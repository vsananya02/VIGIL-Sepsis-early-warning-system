<img width="2878" height="1230" alt="Screenshot 2026-08-07 133716" src="https://github.com/user-attachments/assets/9d08fc97-14a9-47c5-add2-1d08ef492415" />**VIGIL: Sepsis Early-Warning System**



Clinical AI for high-stakes reliability. A Temporal Fusion Transformer system for early sepsis detection in ICU settings, built for interpretability and honest external validation.



**Overview**



VIGIL flags high-risk sepsis patients ahead of clinical onset, giving a window for intervention on a condition where timing changes outcomes. Trained on large-scale ICU data (MIMIC-IV, eICU-CRD) and validated across 208 hospitals it never saw during training.



* Sepsis kills roughly 1 in 3 patients without early intervention
* Rare-event problem: \~3% of ICU patients develop sepsis → false alarms have to be controlled, not just sensitivity
* Every prediction is auditable - SQLite event log, model card, explicit limitations



**Key** **Results**

External Validation - 200,000 patients, 208 hospitals (eICU-CRD)

*Metric*	                                   *Value*

AUROC	                                   0.8521

Calibration (ECE)	                   0.093 → 0.009 after isotonic calibration

Sensitivity	                           83.7%

Specificity	                           74.4%

Median lead time	                   15h (patients with ≥24h prior ICU history)

Hour-level PPV                           \~9%



Lead time and sensitivity are measured only on patients with at least 24 hours of history — the model needs a full day of vitals to make its first prediction, so scoring earlier than that silently pads the input with zeros and inflates the number. That bug was caught and fixed before these figures were reported; see Data Quality Audits below.



**Head-to-head vs. SIRS**



SIRS is the bedside screen hospitals actually use. Benchmarked on identical patients, identical hours, identical alarm rule (2 consecutive hours above threshold):

&#x09;                                                         *VIGIL	SIRS*

Sepsis caught (overall)	                        83.7%	51.9%

Sepsis caught —                *normal WBC*	82.6%	36.9%

Sepsis caught —                      *overt*	85.4%	72.9%

False alarm rate	                               25.6%	26.7%

Median lead time	                                  19h	20h



Both models raise false alarms at essentially the same rate, so the detection gap isn't bought with more noise. The gap is largest on patients whose white blood cell count stays normal — SIRS requires an abnormal count to fire at all, so it's structurally blind to exactly the presentations VIGIL was built to catch.



**Architecture**



Model: Temporal Fusion Transformer — handles variable-length ICU stays, learns which variables matter at which point in a stay, and produces feature-level attention that's inspectable rather than a black box.



*Temporal features (hourly, 24h window):*  HR, RR, SBP, DBP, SpO2, lactate, WBC, creatinine, temperature

*Static features (at admission):*                  age, gender, baseline creatinine

*Output:*                                                     risk score (0–1) every hour → triage tier



Divergence features - engineered signal for the hardest case: organ function declining while inflammation markers stay normal. This is the pattern behind the WBC-normal result above.

###### 

Serving layer - FastAPI, WebSocket streaming to a live dashboard, 32h ring buffer, SQLite audit trail logging every tier change with hour and reason. A hysteresis state machine cuts alarm flapping (8 raw transitions → 2 displayed) to address the documented reason real early-warning systems get switched off: alarm fatigue.



**Data Pipeline**



MIMIC-IV: 91,791 ICU patients, 7.2M+ hourly records — training and internal validation

eICU-CRD: 200,000+ patients, 208 hospitals — external validation only, never tuned on



**Data Quality Audits**



Two issues were found and corrected during development. Both are reported here rather than left in the numbers above.



1\. Silent label leakage. Timestamp misalignment let the model see information from after sepsis onset during training. Traced to an hour-level leak, fixed by enforcing a strict prediction gap. AUROC dropped 0.98 → 0.91 internally — the honest number, reported instead of the inflated one.



2\. Lead-time padding artifact. The model requires 24 hours of history per prediction; scoring earlier than that pads the missing hours with zeros, which the model reads as "average patient" and can fire on. This inflated measured lead time to 24h. Fixed by refusing to score any window before hour 24 and excluding patients who never reach that threshold before onset. Corrected lead time: 15h.



3\. Calibration. Isotonic regression on the internal validation set. ECE improved 0.093 → 0.009 — a risk score of 50% now corresponds to roughly a 50% observed sepsis rate.



**Demo**



Local replay dashboard: real MIMIC-IV patient trajectories replayed one clinical hour at a time, showing the risk score climb alongside the raw vitals a clinician would actually see.



Public demo (synthetic\_cohort.json) uses four physiology-driven synthetic patients — no real patient data — generated to avoid the trajectories being tuned to flatter the model. Scores are reported as produced, not adjusted after the fact. See make\_synthetic\_cohort.py.



replay\_cohort.json (real MIMIC-IV patients) is excluded from this repository under the PhysioNet Data Use Agreement, which prohibits public redistribution.




**Limitations**

1. Cold start: requires 24h of ICU history; patients who deteriorate on day one receive reduced or no warning
2. PPV \~9% at the operating threshold — most alerts are false positives, which is why the hysteresis layer exists
3. Label is a SOFA-style proxy (creatinine, MAP, lactate thresholds), not confirmed Sepsis-3 — and three of the label's components are also model inputs. Mortality prediction (AUROC 0.726, external) is reported as an independent check against this circularity, since it shares none of the label's terms
4. Retrospective only — no prospective clinical trial
5. Research demonstration, not a medical device — decision-support only, not a replacement for clinical judgment



**Resources**

1. MIMIC-IV: https://mimic.mit.edu (requires PhysioNet credentialing)
2. eICU-CRD: https://eicu-crd.mit.edu
3. Temporal Fusion Transformer: Lim et al., 2021
4. Sepsis-3 Consensus: Singer et al., JAMA 2016



**Contact:** vsananya02@gmail.com

**LinkedIn**:https://www.linkedin.com/in/v-s-ananya-21b32a28a/

