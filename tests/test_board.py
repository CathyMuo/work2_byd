"""CPU tests for the progress board (scanner, campaign, log tail, HTTP)."""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from work2 import board
from work2.common import ROOT, write_json as _write_json


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value)


def make_runs(out):
    pcr = out / "u1" / "zs_llama32_3b" / "pcr"
    write_json(pcr / "run_manifest.json", {"status": "running"})
    write_json(pcr / "workers" / "worker_0" / "run_manifest.json", {"worker_index": 0, "gpu": "0", "status": "running", "expected_states": 10, "recorded_states": 4})
    write_json(pcr / "workers" / "worker_1" / "run_manifest.json", {"worker_index": 1, "gpu": "1", "status": "running", "expected_states": 10, "recorded_states": 6})
    (pcr / "logs").mkdir()
    (pcr / "logs" / "worker_0.log").write_text("line1\nline2\n")
    hr = out / "hr2_ppo"
    write_json(hr / "run_manifest.json", {"status": "running"})
    for phase, rec in (("r1", 5), ("i2", 2)):
        write_json(hr / "workers" / phase / "worker_0.manifest.json", {"worker_index": 0, "status": "completed" if phase == "r1" else "running", "expected": 5, "recorded": rec})
    u2 = out / "u2" / "smoke_reseek_reseek_3b"
    write_json(u2 / "run_manifest.json", {"status": "completed", "smoke": True, "recorded": 10})
    old = out / "u1" / "v03_3b_ppo" / "sources"
    write_json(old / "run_manifest.json", {"status": "running"})
    write_json(old / "workers" / "worker_0" / "run_manifest.json", {"status": "failed", "expected": 3, "recorded": 1, "error": {"message": "boom"}})
    for p in old.rglob("*"):
        os.utime(p, (time.time() - 7200, time.time() - 7200))


class ScannerTests(unittest.TestCase):
    def test_runs_phases_status_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_runs(out)
            roster = out / "roster.json"; write_json(roster, {"agents": [{"label": "zs_llama32_3b"}]})
            s = board.Scanner(out, [roster])
            runs = {r["path"]: r for r in s.runs()}
            self.assertEqual(set(runs), {"u1/zs_llama32_3b/pcr", "hr2_ppo", "u2/smoke_reseek_reseek_3b", "u1/v03_3b_ppo/sources"})
            p = runs["u1/zs_llama32_3b/pcr"]
            self.assertEqual((p["experiment"], p["subject"], p["stage"], p["expected"], p["recorded"]), ("U1x", "zs_llama32_3b", "H1 同状态安慰剂分支", 20, 10))
            h = runs["hr2_ppo"]
            self.assertEqual((h["phase"], h["expected"], h["recorded"], h["phases"]["r1"]["done"]), ("i2", 5, 2, True))
            self.assertEqual(runs["u2/smoke_reseek_reseek_3b"]["experiment"], "U2（smoke）")
            self.assertEqual(runs["u1/v03_3b_ppo/sources"]["status"], "failed")
            self.assertEqual(runs["u1/v03_3b_ppo/sources"]["experiment"], "U1")
            logs = s.log_tail("u1/zs_llama32_3b/pcr")
            self.assertEqual(logs[0]["lines"], ["line1", "line2"])
            self.assertIsNone(s.log_tail("../../etc"))

    def test_stalled_and_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            d = out / "u1" / "a" / "sources"
            write_json(d / "run_manifest.json", {"status": "running"})
            write_json(d / "workers" / "worker_0" / "run_manifest.json", {"status": "running", "expected": 100, "recorded": 10})
            for p in d.rglob("*"):
                os.utime(p, (time.time() - 3600, time.time() - 3600))
            s = board.Scanner(out)
            self.assertEqual(s.runs()[0]["status"], "stalled")
            self.assertAlmostEqual(s.rate("k", 10, 1000.0) or 0, 0)
            self.assertAlmostEqual(s.rate("k", 70, 1060.0 + 1), 60 / 61)

    def test_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs"
            make_runs(out)
            s = board.Scanner(out)
            spec = {"items": [{"name": "a", "run": "outputs/u1/zs_llama32_3b/pcr", "done": "outputs/__never__/x"},
                              {"name": "b", "run": "outputs/u1/none/pcr", "done": "outputs/__never__/y"},
                              {"name": "c", "run": "", "done": "configs/u1x_agents.json"}]}
            got = {c["name"]: c for c in s.campaign(spec)}
            self.assertEqual((got["a"]["state"], got["a"]["recorded"]), ("running", 10))
            self.assertEqual(got["b"]["state"], "pending")
            self.assertEqual(got["c"]["state"], "done")


class ContentTests(unittest.TestCase):
    def test_research_json_complete(self):
        d = json.loads((ROOT / "progress_board" / "research.json").read_text(encoding="utf-8"))
        for key in ("title", "talk_track", "overview", "question", "method", "findings", "negatives", "timeline", "campaign"):
            self.assertIn(key, d)
        for f in d["findings"]:
            if "chart" in f:
                c = f["chart"]
                self.assertTrue(all(len(r["values"]) == len(c["series"]) for r in c["rows"]))
        page = (ROOT / "progress_board" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("http://", page.replace("http://localhost", ""))
        self.assertNotIn("https://", page)


class HttpTests(unittest.TestCase):
    def test_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            make_runs(out)
            server = board.ThreadingHTTPServer(("127.0.0.1", 0), board.make_handler(board.Scanner(out)))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = "http://127.0.0.1:{}".format(server.server_address[1])
            try:
                with mock.patch.object(board, "gpus", return_value=None):
                    live = json.loads(urllib.request.urlopen(base + "/api/live").read())
                self.assertEqual(len(live["runs"]), 4)
                self.assertIn("进度看板", urllib.request.urlopen(base + "/").read().decode("utf-8"))
                self.assertIn("talk_track", urllib.request.urlopen(base + "/research.json").read().decode("utf-8"))
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/api/log?run=../x")
            finally:
                server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
