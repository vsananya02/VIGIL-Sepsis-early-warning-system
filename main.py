"""
VIGIL — main.py
The server. Runs a clock, ticks all patients forward, pushes to browsers.

Run:
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/api/patients
"""
import asyncio, json, os, sqlite3, time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from features import FeatureBuilder
from inference import VigilEngine, LactateTracker
from vitals_source import MimicReplaySource, VigilMonitor
from alarms import AlarmBoard

BUNDLE = os.getenv('VIGIL_BUNDLE', 'VIGIL_DEPLOY_v1.pt')
COHORT = os.getenv('VIGIL_COHORT', 'replay_cohort.json')
TICK_SECONDS = float(os.getenv('VIGIL_TICK', '3'))     # 3 real sec = 1 clinical hour
DB = os.getenv('VIGIL_DB', 'vigil_events.db')


# ============================================================== AUDIT LOG
def init_db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS events (
        wall_time REAL, sim_hour INTEGER, patient_id TEXT,
        from_tier TEXT, to_tier TEXT, reason TEXT, risk REAL)""")
    c.commit(); c.close()


def log_event(sim_hour, pid, frm, to, reason, risk):
    c = sqlite3.connect(DB)
    c.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)",
              (time.time(), sim_hour, pid, frm, to, reason, risk))
    c.commit(); c.close()


# ============================================================ SHARED STATE
class Hub:
    def __init__(self):
        self.monitor = None
        self.alarms = AlarmBoard()
        self.board = {}          # patient_id -> latest result
        self.sockets = set()
        self.running = False

    def snapshot(self):
        rows = sorted(self.board.values(),
                      key=lambda r: r.get('lactate_adjusted_risk', -1), reverse=True)
        return {'sim_hour': self.monitor.clock if self.monitor else 0,
                'tick_seconds': TICK_SECONDS,
                'n_patients': len(rows), 'patients': rows}

    async def broadcast(self, payload):
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)


hub = Hub()


# ================================================================ THE CLOCK
async def clock_loop():
    hub.running = True
    while hub.running:
        await asyncio.sleep(TICK_SECONDS)

        # cohort exhausted -> restart so the demo never dies
        if not hub.monitor.any_active():
            hub.monitor.source.reset()
            hub.monitor = VigilMonitor(BUNDLE, hub.monitor.source)
            hub.alarms = AlarmBoard()
            hub.board.clear()
            await hub.broadcast({'type': 'restart'})
            continue

        for r in hub.monitor.tick():
            pid = r['patient_id']

            if r['status'] == 'MONITORING':
                a = hub.alarms.update(pid, r['tier'], hour=hub.monitor.clock)
                r['displayed_tier'] = a['tier']
                r['alarm'] = a['alarm']
                r['acknowledged'] = a['acknowledged']
                r['pending_tier'] = a.get('pending')
                r['active_alarm'] = (a['tier'] in ('RED','ORANGE')) and not a['acknowledged']
                if a['changed']:
                    log_event(hub.monitor.clock, pid, r['tier'], a['tier'],
                              a['reason'], r['lactate_adjusted_risk'])
            else:
                r['displayed_tier'] = 'PENDING'
                r['alarm'] = False

            hub.board[pid] = r

        await hub.broadcast({'type': 'tick', **hub.snapshot()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    hub.monitor = VigilMonitor(BUNDLE, MimicReplaySource(COHORT))
    task = asyncio.create_task(clock_loop())
    yield
    hub.running = False
    task.cancel()


app = FastAPI(title="VIGIL", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ================================================================ ENDPOINTS
@app.get("/")
def dashboard():
    return FileResponse("dashboard.html")
@app.get("/health")
def health():
    return {"ok": True, "sim_hour": hub.monitor.clock if hub.monitor else 0,
            "patients": len(hub.board)}


@app.get("/api/patients")
def patients():
    """The ward board."""
    return hub.snapshot()


@app.get("/api/patients/{pid}")
def patient(pid: str):
    """One patient, with their full risk trajectory."""
    if pid not in hub.board:
        raise HTTPException(404, "unknown patient")
    st = hub.monitor.states[pid]
    return {**hub.board[pid],
            'risk_history': st.history,
            'hours_seen': st.hours_seen,
            'vitals_history': list(st.buffer),
            'transitions': hub.alarms.states[pid].transitions if pid in hub.alarms.states else []}


@app.post("/api/patients/{pid}/acknowledge")
def acknowledge(pid: str):
    """Nurse silences the alarm. Re-fires only if the patient worsens."""
    hub.alarms.acknowledge(pid)
    if pid in hub.board:
        hub.board[pid]['acknowledged'] = True
        hub.board[pid]['alarm'] = False
    return {"ok": True, "patient_id": pid}


@app.get("/api/model-card")
def model_card():
    import torch
    b = torch.load(BUNDLE, map_location='cpu', weights_only=False)
    return {'metrics': b['metrics'], 'tiers': b['tiers'],
            'intended_use': b['intended_use'],
            'created': b['created_utc'],
            'limitations': [
                "Requires 24h of ICU history; patients deteriorating within "
                "their first day receive reduced or no lead time.",
                "Calibration bin 0.3-0.4 is +5.4pp optimistic (n=1401).",
                "eICU external metrics were computed without the calibrator applied.",
            ]}


@app.get("/api/audit")
def audit(limit: int = 100):
    c = sqlite3.connect(DB)
    rows = c.execute("SELECT * FROM events ORDER BY wall_time DESC LIMIT ?",
                     (limit,)).fetchall()
    c.close()
    cols = ['wall_time', 'sim_hour', 'patient_id', 'from_tier', 'to_tier', 'reason', 'risk']
    return [dict(zip(cols, r)) for r in rows]


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    hub.sockets.add(websocket)
    await websocket.send_json({'type': 'snapshot', **hub.snapshot()})
    try:
        while True:
            await websocket.receive_text()      # keepalive
    except WebSocketDisconnect:
        hub.sockets.discard(websocket)
