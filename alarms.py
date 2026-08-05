"""
VIGIL — alarms.py

Stops tier flapping. Raw model output can cross a threshold every hour;
a nurse should not see a card strobing all night. Alarm fatigue is the
documented reason real early-warning systems get switched off.

Rules
  1. Escalate only after ESCALATE_HOURS consecutive hours at the higher tier.
  2. UNLESS the jump is >= JUMP_TIERS levels — that fires immediately.
  3. De-escalate only after DEESCALATE_HOURS consecutive hours lower.
  4. Acknowledged alarms stay quiet unless the patient gets worse.
"""

TIERS = ['GREEN', 'YELLOW', 'ORANGE', 'RED']
RANK = {t: i for i, t in enumerate(TIERS)}

ESCALATE_HOURS   = 2      # sustained rise before we act
DEESCALATE_HOURS = 3      # slower to relax than to worry
JUMP_TIERS       = 2      # a 2-tier leap bypasses the wait


class AlarmState:
    """One patient's alarm. Feed it the raw tier each hour."""

    def __init__(self, pid):
        self.pid = pid
        self.displayed = 'GREEN'      # what the dashboard shows
        self.candidate = None         # tier we're currently counting toward
        self.streak = 0               # consecutive hours at candidate
        self.acknowledged = False
        self.ack_at_rank = -1         # tier level when acknowledged
        self.transitions = []         # audit trail

    def update(self, raw_tier, hour=None):
        cur, new = RANK[self.displayed], RANK[raw_tier]

        # --- rule 2: big jump, act now ---
        if new - cur >= JUMP_TIERS:
            return self._commit(raw_tier, hour, 'jump')

        # --- no change wanted ---
        if new == cur:
            self.candidate, self.streak = None, 0
            return self._quiet()

        # --- counting toward a change ---
        if raw_tier != self.candidate:
            self.candidate, self.streak = raw_tier, 1
        else:
            self.streak += 1

        need = ESCALATE_HOURS if new > cur else DEESCALATE_HOURS
        if self.streak >= need:
            return self._commit(raw_tier, hour, 'sustained')

        return self._quiet(pending=raw_tier, hours_to_go=need - self.streak)

    # ------------------------------------------------------------------
    def _commit(self, tier, hour, reason):
        was = self.displayed
        self.displayed = tier
        self.candidate, self.streak = None, 0
        escalating = RANK[tier] > RANK[was]

        # a worsening patient re-alarms even if previously acknowledged
        if escalating and RANK[tier] > self.ack_at_rank:
            self.acknowledged = False

        self.transitions.append({'hour': hour, 'from': was, 'to': tier,
                                 'reason': reason})
        return {'tier': tier, 'changed': True, 'escalating': escalating,
                'reason': reason,
                'alarm': escalating and not self.acknowledged,
                'acknowledged': self.acknowledged}

    def _quiet(self, pending=None, hours_to_go=0):
        return {'tier': self.displayed, 'changed': False, 'escalating': False,
                'reason': None, 'alarm': False,
                'acknowledged': self.acknowledged,
                'pending': pending, 'hours_to_go': hours_to_go}

    def acknowledge(self):
        self.acknowledged = True
        self.ack_at_rank = RANK[self.displayed]


class AlarmBoard:
    """All patients' alarms."""

    def __init__(self):
        self.states = {}

    def update(self, pid, raw_tier, hour=None):
        if pid not in self.states:
            self.states[pid] = AlarmState(pid)
        return self.states[pid].update(raw_tier, hour)

    def acknowledge(self, pid):
        if pid in self.states:
            self.states[pid].acknowledge()

    def audit(self):
        return {pid: s.transitions for pid, s in self.states.items()}


# ==============================================================================
# ACCEPTANCE TEST — raw tiers flap; the board must not.
# ==============================================================================
def _tier_of(risk):
    if   risk >= 0.75: return 'RED'
    elif risk >= 0.50: return 'ORANGE'
    elif risk >= 0.25: return 'YELLOW'
    return 'GREEN'


def test_alarms(risk_sequence=None):
    # a patient hovering right on the RED line — the worst case for flapping
    if risk_sequence is None:
        risk_sequence = [0.72, 0.77, 0.73, 0.78, 0.74, 0.76, 0.71, 0.79,
                         0.80, 0.82, 0.85, 0.88, 0.90, 0.40, 0.38, 0.36]

    raw = [_tier_of(r) for r in risk_sequence]
    raw_flips = sum(1 for a, b in zip(raw, raw[1:]) if a != b)

    board = AlarmBoard()
    shown = []
    for h, r in enumerate(risk_sequence):
        res = board.update('ICU-01', _tier_of(r), hour=h)
        shown.append(res['tier'])
    smooth_flips = sum(1 for a, b in zip(shown, shown[1:]) if a != b)

    print(f"  raw tier changes    : {raw_flips}")
    print(f"  after hysteresis    : {smooth_flips}")
    print(f"  transitions logged  : {board.audit()['ICU-01']}")
    print(f"\n  {'PASS — flapping suppressed' if smooth_flips < raw_flips else 'FAIL'}")
    return smooth_flips < raw_flips


# In Colab:  test_alarms()
