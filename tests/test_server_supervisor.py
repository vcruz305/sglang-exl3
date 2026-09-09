import json
import os
import subprocess
import sys
import socket
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "server_supervisor.py"


def run_supervisor(*args, timeout=10):
    return subprocess.run(
        [sys.executable, str(SUPERVISOR), *map(str, args)],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class ServerSupervisorTests(unittest.TestCase):
    @staticmethod
    def pid_alive(pid):
        if os.name == "nt":
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
            return str(pid) in result.stdout
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def start_args(self, td, command, **extra):
        status = Path(td) / "server.status.json"
        log = Path(td) / "server.log"
        meminfo = Path(td) / "meminfo"
        if "meminfo" not in extra:
            # 125 GiB available is a safe synthetic host for local admission
            # tests; the remote gate still reads the real /proc/meminfo.
            meminfo.write_text("MemAvailable:       131072000 kB\n")
            extra["meminfo"] = meminfo
        args = [
            "start",
            "--run-id",
            "test-run",
            "--status",
            status,
            "--log",
            log,
            "--startup-deadline",
            extra.pop("startup_deadline", 2),
            "--interval",
            extra.pop("interval", 0.05),
            "--reserve-gib",
            extra.pop("reserve_gib", 20),
            "--model-gib",
            extra.pop("model_gib", 0),
            "--draft-gib",
            extra.pop("draft_gib", 0),
            "--kv-gib",
            extra.pop("kv_gib", 0),
            "--graph-gib",
            extra.pop("graph_gib", 0),
            "--scratch-gib",
            extra.pop("scratch_gib", 0),
        ]
        for key, flag in (("ready_file", "--ready-file"), ("ready_url", "--ready-url"), ("meminfo", "--meminfo"), ("guard_gib", "--guard-gib")):
            if key in extra:
                args.extend([flag, extra.pop(key)])
        args.extend(["--", *command])
        return args, status, log

    def test_ready_run_has_durable_status_and_owned_stop(self):
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "ready"
            code = "import pathlib,time; pathlib.Path(%r).touch(); time.sleep(30)" % str(ready)
            args, status, log = self.start_args(td, [sys.executable, "-c", code], ready_file=ready)
            proc = subprocess.Popen([sys.executable, str(SUPERVISOR), *map(str, args)])
            deadline = time.time() + 5
            while time.time() < deadline:
                if status.exists():
                    try:
                        if json.loads(status.read_text())["state"] == "READY":
                            break
                    except (PermissionError, json.JSONDecodeError):
                        pass
                time.sleep(0.05)
            self.assertTrue(status.exists())
            self.assertEqual(json.loads(status.read_text())["state"], "READY")
            stopped = run_supervisor("stop", "--status", status, "--grace", 1)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            proc.wait(timeout=5)
            self.assertEqual(json.loads(status.read_text())["state"], "STOPPED")
            self.assertTrue(log.exists())

    def test_http_readiness_gate(self):
        with tempfile.TemporaryDirectory() as td:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.close()
            code = (
                "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                "class H(BaseHTTPRequestHandler):\n"
                "  def do_GET(self):\n"
                "    self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
                "  def log_message(self,*a): pass\n"
                f"HTTPServer(('127.0.0.1',{port}),H).serve_forever()"
            )
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], startup_deadline=3, interval=0.05, ready_url=f"http://127.0.0.1:{port}/health")
            supervisor = subprocess.Popen([sys.executable, str(SUPERVISOR), *map(str, args)])
            deadline = time.time() + 5
            state = None
            while time.time() < deadline:
                try:
                    state = json.loads(status.read_text())["state"]
                except (PermissionError, json.JSONDecodeError, FileNotFoundError):
                    pass
                if state == "READY":
                    break
                time.sleep(0.05)
            self.assertEqual(state, "READY")
            stopped = run_supervisor("stop", "--status", status, "--grace", 1)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            supervisor.wait(timeout=5)

    def test_startup_timeout_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as td:
            code = "import time; time.sleep(30)"
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], startup_deadline=0.2, ready_file=Path(td) / "never")
            result = run_supervisor(*args)
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertEqual(json.loads(status.read_text())["state"], "STARTUP_TIMEOUT")

    def test_immediate_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            args, status, _ = self.start_args(td, [sys.executable, "-c", "raise SystemExit(7)"], ready_file=Path(td) / "never")
            result = run_supervisor(*args)
            self.assertEqual(result.returncode, 42)
            data = json.loads(status.read_text())
            self.assertEqual(data["state"], "STARTUP_FAILED")
            self.assertEqual(data["returncode"], 7)

    def test_admission_refusal_happens_before_child(self):
        with tempfile.TemporaryDirectory() as td:
            meminfo = Path(td) / "meminfo"
            meminfo.write_text("MemAvailable:       1000 kB\n")
            args, status, _ = self.start_args(td, [sys.executable, "-c", "raise SystemExit(99)"], meminfo=meminfo, model_gib=1, guard_gib=0)
            result = run_supervisor(*args)
            self.assertEqual(result.returncode, 41)
            self.assertEqual(json.loads(status.read_text())["state"], "MEMORY_ADMISSION_REFUSED")

    def test_memory_guard_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as td:
            meminfo = Path(td) / "meminfo"
            meminfo.write_text("MemAvailable:       104857600 kB\n")
            code = "import time; time.sleep(30)"
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], meminfo=meminfo, guard_gib=101, reserve_gib=0)
            # reserve_gib is not exposed by the helper; admission still passes
            # because the test has no expected peak and the guard trips first.
            args.extend([])
            result = run_supervisor(*args)
            self.assertEqual(result.returncode, 41)
            self.assertEqual(json.loads(status.read_text())["state"], "MEMORY_GUARD_STOPPED")

    def test_memory_telemetry_loss_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "ready"
            meminfo = Path(td) / "meminfo"
            code = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], ready_file=ready)
            supervisor = subprocess.Popen([sys.executable, str(SUPERVISOR), *map(str, args)])
            deadline = time.time() + 5
            state = None
            while time.time() < deadline:
                try:
                    state = json.loads(status.read_text())["state"]
                except (PermissionError, json.JSONDecodeError, FileNotFoundError):
                    pass
                if state == "READY":
                    break
                time.sleep(0.05)
            self.assertEqual(state, "READY")
            meminfo.unlink()
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    state = json.loads(status.read_text())["state"]
                except (PermissionError, json.JSONDecodeError, FileNotFoundError):
                    pass
                if state == "MEMORY_TELEMETRY_LOST":
                    break
                time.sleep(0.05)
            supervisor.wait(timeout=5)
            self.assertEqual(state, "MEMORY_TELEMETRY_LOST")

    def test_stale_pid_refuses_to_signal_unrelated_process(self):
        with tempfile.TemporaryDirectory() as td:
            sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
            status = Path(td) / "server.status.json"
            status.write_text(json.dumps({"state": "READY", "child": {"pid": sleeper.pid, "starttime": "not-the-real-start"}}))
            result = run_supervisor("stop", "--status", status)
            self.assertEqual(result.returncode, 43)
            self.assertEqual(json.loads(status.read_text())["state"], "STALE_PID_REFUSED")
            self.assertIsNone(sleeper.poll())
            sleeper.terminate()
            sleeper.wait(timeout=3)

    def test_missing_status_is_a_safe_noop(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_supervisor("stop", "--status", Path(td) / "missing.status.json")
            self.assertEqual(result.returncode, 0)
            self.assertIn("No status file", result.stdout)

    def test_owned_descendant_cleanup(self):
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "ready"
            child_pid = Path(td) / "child.pid"
            code = (
                "import pathlib,subprocess,sys,time; "
                f"c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"pathlib.Path({str(child_pid)!r}).write_text(str(c.pid)); "
                f"pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
            )
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], ready_file=ready)
            supervisor = subprocess.Popen([sys.executable, str(SUPERVISOR), *map(str, args)])
            deadline = time.time() + 5
            while time.time() < deadline and not child_pid.exists():
                time.sleep(0.05)
            self.assertTrue(child_pid.exists())
            child = int(child_pid.read_text())
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    if json.loads(status.read_text())["state"] == "READY":
                        break
                except (PermissionError, json.JSONDecodeError, FileNotFoundError):
                    pass
                time.sleep(0.05)
            stopped = run_supervisor("stop", "--status", status, "--grace", 1)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            supervisor.wait(timeout=5)
            self.assertFalse(self.pid_alive(child), "owned descendant survived cleanup")

    @unittest.skipIf(os.name == "nt", "Windows TerminateProcess bypasses Python signal handlers; exercised on Linux target")
    def test_supervisor_signal_cleans_owned_child(self):
        with tempfile.TemporaryDirectory() as td:
            ready = Path(td) / "ready"
            code = f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(30)"
            args, status, _ = self.start_args(td, [sys.executable, "-c", code], ready_file=ready)
            supervisor = subprocess.Popen([sys.executable, str(SUPERVISOR), *map(str, args)])
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    if json.loads(status.read_text())["state"] == "READY":
                        break
                except (PermissionError, json.JSONDecodeError, FileNotFoundError):
                    pass
                time.sleep(0.05)
            supervisor.terminate()
            supervisor.wait(timeout=5)
            self.assertEqual(json.loads(status.read_text())["state"], "STOPPED_BY_SUPERVISOR")


if __name__ == "__main__":
    unittest.main()
