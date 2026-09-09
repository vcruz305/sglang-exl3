#!/usr/bin/env python3
"""Bounded, ownership-safe supervision for one SGLang server run.

The supervisor deliberately does not discover or kill processes by name.  A
run is owned by its leader PID, proc start-time, and dedicated process group.
The status file is durable enough to reconcile a dropped SSH session without
assuming that a launcher PID implies a ready server.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


EXIT_ADMISSION = 41
EXIT_STARTUP = 42
EXIT_STALE = 43
EXIT_USAGE = 64
GiB = 1024**3
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows can briefly hold the destination while the separate
            # stop command reconciles the same durable receipt.
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def proc_starttime(pid: int) -> str | None:
    if os.name == "nt":
        # Portable creation-time identity for local dummy-process tests and
        # Windows staging.  Production Spark runs use /proc starttime.
        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        try:
            ok = kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user))
            if not ok:
                return None
            return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel32.CloseHandle(handle)
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    # comm may contain spaces and ')' so split after the final ')' before the
    # state field; starttime is field 22, i.e. offset 19 after state.
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def proc_pgid(pid: int) -> int | None:
    if os.name == "nt":
        return None
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def identity(pid: int) -> dict[str, Any] | None:
    start = proc_starttime(pid)
    if start is None:
        return None
    return {"pid": pid, "starttime": start, "pgid": proc_pgid(pid), "cmdline": proc_cmdline(pid)}


def owned(meta: dict[str, Any]) -> bool:
    pid = int(meta.get("pid", -1))
    current = identity(pid)
    return bool(current and str(current["starttime"]) == str(meta.get("starttime")))


def mem_available(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                value = int(parts[1])
                return value * (1024 if len(parts) < 3 or parts[2].lower() == "kb" else 1)
    return None


def admission(args: argparse.Namespace) -> dict[str, Any]:
    components = {
        "model_gib": args.model_gib,
        "draft_gib": args.draft_gib,
        "kv_gib": args.kv_gib,
        "graph_gib": args.graph_gib,
        "scratch_gib": args.scratch_gib,
    }
    expected = sum(float(v) for v in components.values())
    available = mem_available(Path(args.meminfo))
    result = {
        "checked_at": now(),
        "meminfo": args.meminfo,
        "available_bytes": available,
        "available_gib": None if available is None else available / GiB,
        "components_gib": components,
        "expected_peak_gib": expected,
        "reserve_gib": float(args.reserve_gib),
        "admitted": False,
        "reason": "meminfo unavailable",
    }
    if available is not None:
        required = (expected + float(args.reserve_gib)) * GiB
        result["required_bytes"] = required
        result["admitted"] = available >= required
        result["reason"] = "ok" if result["admitted"] else "conservative reserve would be breached"
    return result


def update(status_path: Path, status: dict[str, Any], state: str, **extra: Any) -> None:
    status["state"] = state
    status["updated_at"] = now()
    status.update(extra)
    write_json(status_path, status)


def ready(args: argparse.Namespace) -> bool:
    if args.ready_file:
        return Path(args.ready_file).exists()
    if args.ready_url:
        try:
            with urllib.request.urlopen(args.ready_url, timeout=1.0) as response:
                return 200 <= response.status < 400
        except (OSError, urllib.error.URLError):
            return False
    return False


def signal_owned(meta: dict[str, Any], sig: int) -> bool:
    if not owned(meta):
        return False
    pgid = meta.get("pgid")
    try:
        if os.name == "nt":
            # Windows has no portable process-group signal API.  PID was
            # identity-checked above; /T descends only from that verified
            # leader, and never performs name-based discovery.  This is the
            # owned-tree fallback for both graceful and forced phases.
            result = subprocess.run(["taskkill", "/PID", str(meta["pid"]), "/T", "/F"], capture_output=True)
            return result.returncode == 0
        if pgid is not None and int(pgid) > 1:
            os.killpg(int(pgid), sig)
        else:
            os.kill(int(meta["pid"]), sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def wait_dead(pid: int, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if proc_starttime(pid) is None:
            return True
        time.sleep(0.05)
    return proc_starttime(pid) is None


def stop_owned(status_path: Path, grace: float = 5.0) -> int:
    if not status_path.exists():
        print(f"No status file: {status_path}")
        return 0
    status = read_json(status_path)
    child = status.get("child") or {}
    if not child or not owned(child):
        update(status_path, status, "STALE_PID_REFUSED", refusal="leader identity/start-time mismatch")
        print("Refused to signal an absent, stale, or reused PID")
        return EXIT_STALE
    signaled = signal_owned(child, signal.SIGTERM)
    dead = wait_dead(int(child["pid"]), grace)
    if not dead:
        signal_owned(child, signal.SIGKILL)
        dead = wait_dead(int(child["pid"]), 2.0)
    update(status_path, status, "STOPPED" if dead else "STOP_TIMEOUT", signal_sent=signaled)
    print(f"owned cleanup state={status['state']} pid={child['pid']}")
    return 0 if dead else EXIT_STARTUP


def run(args: argparse.Namespace) -> int:
    status_path = Path(args.status)
    if status_path.exists():
        try:
            old = read_json(status_path)
            old_child = old.get("child") or {}
            if old_child and owned(old_child):
                print("Refusing to replace an existing owned run", file=sys.stderr)
                return EXIT_STALE
            update(status_path, old, "STALE_PID_REFUSED", refusal="old status did not own a live matching process")
        except (OSError, ValueError, TypeError):
            status_path.unlink(missing_ok=True)

    admit = admission(args)
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at": now(),
        "command": args.command,
        "log": str(Path(args.log).resolve()),
        "status_path": str(status_path.resolve()),
        "admission": admit,
        "supervisor_pid": os.getpid(),
    }
    if not admit["admitted"]:
        update(status_path, status, "MEMORY_ADMISSION_REFUSED")
        print(json.dumps(admit, sort_keys=True), file=sys.stderr)
        return EXIT_ADMISSION

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        child = subprocess.Popen(
            args.command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=args.cwd or None,
            env=os.environ.copy(),
        )
        child_meta = identity(child.pid)
        if child_meta is None:
            update(status_path, status, "STARTUP_FAILED", error="child identity unavailable")
            return EXIT_STARTUP
        status["child"] = child_meta
        status["readiness"] = {"url": args.ready_url, "file": args.ready_file}
        update(status_path, status, "STARTING")

        # SSH loss must not restart or stop the owned run.  An explicit
        # supervisor termination, however, performs bounded owned cleanup.
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)

        def handle_supervisor_signal(signum, _frame):
            signal_owned(child_meta, signal.SIGTERM)
            wait_dead(child.pid, 2.0)
            if proc_starttime(child.pid) is not None:
                signal_owned(child_meta, KILL_SIGNAL)
            update(status_path, status, "STOPPED_BY_SUPERVISOR", signal=int(signum))
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, handle_supervisor_signal)

        deadline = time.monotonic() + float(args.startup_deadline)
        while time.monotonic() < deadline:
            if child.poll() is not None:
                update(status_path, status, "STARTUP_FAILED", returncode=child.returncode)
                return EXIT_STARTUP
            available = mem_available(Path(args.meminfo))
            status["last_memory"] = {"at": now(), "available_bytes": available, "available_gib": None if available is None else available / GiB}
            if available is None:
                signal_owned(child_meta, signal.SIGTERM)
                wait_dead(child.pid, 2.0)
                if proc_starttime(child.pid) is not None:
                    signal_owned(child_meta, KILL_SIGNAL)
                update(status_path, status, "MEMORY_TELEMETRY_LOST")
                return EXIT_ADMISSION
            if available is not None and available < float(args.guard_gib) * GiB:
                signal_owned(child_meta, signal.SIGTERM)
                wait_dead(child.pid, 2.0)
                if proc_starttime(child.pid) is not None:
                    signal_owned(child_meta, KILL_SIGNAL)
                update(status_path, status, "MEMORY_GUARD_STOPPED", guard_gib=float(args.guard_gib))
                return EXIT_ADMISSION
            if ready(args):
                update(status_path, status, "READY", ready_at=now())
                break
            time.sleep(float(args.interval))
        else:
            signal_owned(child_meta, signal.SIGTERM)
            wait_dead(child.pid, 2.0)
            if proc_starttime(child.pid) is not None:
                signal_owned(child_meta, KILL_SIGNAL)
            update(status_path, status, "STARTUP_TIMEOUT")
            return EXIT_STARTUP

        # Continue monitoring the owned run.  SSH loss does not restart it.
        while child.poll() is None:
            available = mem_available(Path(args.meminfo))
            status["last_memory"] = {"at": now(), "available_bytes": available, "available_gib": None if available is None else available / GiB}
            if available is None:
                signal_owned(child_meta, signal.SIGTERM)
                wait_dead(child.pid, 2.0)
                if proc_starttime(child.pid) is not None:
                    signal_owned(child_meta, KILL_SIGNAL)
                update(status_path, status, "MEMORY_TELEMETRY_LOST")
                return EXIT_ADMISSION
            if available is not None and available < float(args.guard_gib) * GiB:
                signal_owned(child_meta, signal.SIGTERM)
                wait_dead(child.pid, 2.0)
                if proc_starttime(child.pid) is not None:
                    signal_owned(child_meta, KILL_SIGNAL)
                update(status_path, status, "MEMORY_GUARD_STOPPED", guard_gib=float(args.guard_gib))
                return EXIT_ADMISSION
            write_json(status_path, status)
            time.sleep(float(args.interval))
        update(status_path, status, "EXITED", returncode=child.returncode, exited_at=now())
        return int(child.returncode or 0)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start")
    start.add_argument("--status", required=True)
    start.add_argument("--log", required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--cwd")
    start.add_argument("--ready-url")
    start.add_argument("--ready-file")
    start.add_argument("--startup-deadline", type=float, default=900)
    start.add_argument("--interval", type=float, default=2)
    start.add_argument("--reserve-gib", type=float, default=20)
    start.add_argument("--guard-gib", type=float, default=20)
    start.add_argument("--model-gib", type=float, default=0)
    start.add_argument("--draft-gib", type=float, default=0)
    start.add_argument("--kv-gib", type=float, default=0)
    start.add_argument("--graph-gib", type=float, default=0)
    start.add_argument("--scratch-gib", type=float, default=0)
    start.add_argument("--meminfo", default="/proc/meminfo")
    start.add_argument("command", nargs=argparse.REMAINDER)
    stop = sub.add_parser("stop")
    stop.add_argument("--status", required=True)
    stop.add_argument("--grace", type=float, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "start":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser().error("start requires a command after --")
        return run(args)
    return stop_owned(Path(args.status), args.grace)


if __name__ == "__main__":
    raise SystemExit(main())
