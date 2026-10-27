"""Progress board: a read-only local web page for factory runs plus the research briefing.

python -m work2.board [--host 127.0.0.1] [--port 8765] [--outputs outputs]

Standard library only (the factory has no internet; the page loads nothing external).
It never writes anywhere and only reads:
  progress_board/index.html, progress_board/research.json   page and briefing content (edited at home)
  <outputs>/**/run_manifest.json, workers/*/run_manifest.json, workers/<phase>/worker_*.manifest.json
  <run>/logs/*.log (tail only; the run must be one the scanner found)
  nvidia-smi (optional, 5 s timeout)
"""
import argparse
import collections
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import ROOT

PAGE_DIR = ROOT / "progress_board"
STALL_SECONDS = 20 * 60
STAGE_NAMES = {"sources": "源轨迹（贪心作答）", "pcr": "H1 同状态安慰剂分支", "ct": "H3 信用真值采样",
               "pcr_plan": "PCR 计划", "ct_plan": "CT 计划"}


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def roster_labels(path):
    data = load(path) or {}
    return {a["label"] for a in data.get("agents", [])}


def describe(rel, u1x):
    """(experiment, subject, stage) shown on the page for a run directory relative to outputs/."""
    parts = rel.split("/")
    if parts[0] == "u1" and len(parts) >= 3:
        return ("U1x" if parts[1] in u1x else "U1", parts[1], STAGE_NAMES.get(parts[2], parts[2]))
    if parts[0] == "u2" and len(parts) >= 2:
        name = parts[1]
        smoke = name.startswith("smoke_")
        name = name[6:] if smoke else name
        method, _, label = name.partition("_")
        return ("U2" + ("（smoke）" if smoke else ""), label, {"reseek": "ReSeek judge 审计", "mr": "MR-Search 反思审计"}.get(method, method))
    if parts[0].startswith("hr2"):
        return ("HR2", parts[0].replace("hr2_", ""), "交接重放")
    if parts[0].startswith("hr1"):
        return ("HR1", parts[0], "交接重放")
    return (parts[0].upper(), "/".join(parts[1:]) or parts[0], "")


def worker_manifests(run_dir):
    found = sorted(run_dir.glob("workers/worker_*/run_manifest.json")) + sorted(run_dir.glob("workers/*/worker_*.manifest.json"))
    out = []
    for path in found:
        m = load(path)
        if not isinstance(m, dict):
            continue
        phase = path.parent.name if path.name.endswith(".manifest.json") else None
        out.append({"phase": phase or m.get("phase"), "index": m.get("worker_index"), "gpu": m.get("gpu"),
                    "status": m.get("status"), "expected": m.get("expected", m.get("expected_states")),
                    "recorded": m.get("recorded", m.get("recorded_states")), "mtime": path.stat().st_mtime,
                    "error": (m.get("error") or {}).get("message") if isinstance(m.get("error"), dict) else m.get("error")})
    return out


def latest_mtime(run_dir):
    newest = 0.0
    for p in run_dir.rglob("*"):
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:
            pass
    return newest


class Scanner:
    def __init__(self, outputs, rosters=()):
        self.outputs = Path(outputs)
        self.u1x = set().union(*(roster_labels(r) for r in rosters)) if rosters else set()
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=120))
        self.lock = threading.Lock()

    def run_dirs(self):
        if not self.outputs.is_dir():
            return []
        dirs = []
        for path in self.outputs.rglob("run_manifest.json"):
            rel = path.parent.relative_to(self.outputs).as_posix()
            if "/workers/" in "/" + rel + "/" or rel.count("/") > 4:
                continue
            dirs.append(path.parent)
        return dirs

    def rate(self, key, recorded, now):
        with self.lock:
            h = self.history[key]
            if recorded is not None and (not h or h[-1][1] != recorded or now - h[-1][0] > 60):
                h.append((now, recorded))
            window = [x for x in h if now - x[0] <= 15 * 60]
        if len(window) >= 2 and window[-1][0] > window[0][0] and window[-1][1] > window[0][1]:
            return (window[-1][1] - window[0][1]) / (window[-1][0] - window[0][0])
        return None

    def runs(self):
        now = time.time()
        rows = []
        for run_dir in self.run_dirs():
            rel = run_dir.relative_to(self.outputs).as_posix()
            top = load(run_dir / "run_manifest.json") or {}
            workers = worker_manifests(run_dir)
            phases = collections.OrderedDict()
            for w in workers:
                p = phases.setdefault(w["phase"] or "", {"expected": 0, "recorded": 0, "statuses": []})
                p["expected"] += w["expected"] or 0
                p["recorded"] += w["recorded"] or 0
                p["statuses"].append(w["status"])
            current = next((k for k, p in phases.items() if any(s != "completed" for s in p["statuses"])), None)
            focus = phases.get(current) if current is not None else (list(phases.values())[-1] if phases else None)
            expected = focus["expected"] if focus else top.get("planned", top.get("recorded"))
            recorded = focus["recorded"] if focus else top.get("recorded", top.get("recorded_states"))
            status = top.get("status") or "unknown"
            if any(w["status"] == "failed" for w in workers):
                status = "failed"
            mtime = latest_mtime(run_dir)
            if status == "running" and now - mtime > STALL_SECONDS:
                status = "stalled"
            speed = self.rate(rel + "|" + (current or ""), recorded, now) if status == "running" else None
            eta = (expected - recorded) / speed if speed and expected is not None and recorded is not None else None
            exp, subject, stage = describe(rel, self.u1x)
            rows.append({"path": rel, "experiment": exp, "subject": subject, "stage": stage, "status": status,
                         "phase": current, "phases": {k: {"expected": v["expected"], "recorded": v["recorded"],
                                                           "done": all(s == "completed" for s in v["statuses"])}
                                                       for k, v in phases.items()},
                         "expected": expected, "recorded": recorded, "per_minute": speed * 60 if speed else None,
                         "eta_seconds": eta, "updated": mtime, "smoke": bool(top.get("smoke")),
                         "workers": workers, "error": top.get("error") if isinstance(top.get("error"), str) else None})
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return rows

    def campaign(self, spec):
        """Pipeline status of the items listed in research.json (each item: name, run, done)."""
        runs = {r["path"]: r for r in self.runs()}
        out = []
        under = lambda p: self.outputs / Path(p).relative_to("outputs") if p.startswith("outputs/") else ROOT / p
        for item in (spec or {}).get("items", []):
            done = under(item["done"]).exists() if item.get("done") else False
            run = runs.get(Path(item["run"]).relative_to("outputs").as_posix()) if item.get("run", "").startswith("outputs/") else None
            if done:
                state = "done"
            elif run:
                state = run["status"]
            else:
                state = "pending"
            out.append({"name": item["name"], "state": state, "expected": run and run["expected"],
                        "recorded": run and run["recorded"], "eta_seconds": run and run["eta_seconds"]})
        return out

    def log_tail(self, rel, lines=40):
        if rel not in {r["path"] for r in self.runs()}:
            return None
        logs = sorted((self.outputs / rel / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in logs[:4]:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16000))
                text = f.read().decode("utf-8", "replace").splitlines()[-lines:]
            out.append({"file": path.name, "lines": text})
        return out


def gpus():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    out = []
    for line in res.stdout.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 5:
            out.append({"index": p[0], "name": p[1], "util": float(p[2]), "mem_used": float(p[3]), "mem_total": float(p[4])})
    return out


def make_handler(scanner):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def json(self, value):
            self.send(200, json.dumps(value, ensure_ascii=False), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self.send(200, (PAGE_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/research.json":
                self.send(200, (PAGE_DIR / "research.json").read_bytes(), "application/json; charset=utf-8")
            elif url.path == "/api/live":
                spec = (load(PAGE_DIR / "research.json") or {}).get("campaign")
                self.json({"now": time.time(), "runs": scanner.runs(), "gpus": gpus(), "campaign": scanner.campaign(spec),
                           "outputs": str(scanner.outputs)})
            elif url.path == "/api/log":
                tail = scanner.log_tail(q.get("run", [""])[0])
                self.json(tail) if tail is not None else self.send(404, "unknown run", "text/plain; charset=utf-8")
            else:
                self.send(404, "not found", "text/plain; charset=utf-8")

        def log_message(self, *args):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only on a trusted internal network")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    a = p.parse_args()
    scanner = Scanner(a.outputs, [ROOT / "configs/u1x_agents.json"])
    server = ThreadingHTTPServer((a.host, a.port), make_handler(scanner))
    print("Progress board on http://{}:{}/ (outputs: {}); Ctrl-C to stop.".format(a.host, a.port, a.outputs), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
