"""Spawn the headless Electron driver (real app code, throwaway profile) and talk to it over stdin/stdout."""
import json
import os
import queue
import shutil
import subprocess
import threading
import time

from .server_manager import descendants, verify_process_tree_gone
from .util import HARNESS_ROOT, FORBIDDEN_PORTS, HARNESS_PORTS

SENTINEL = "@@HF@@"
DRIVER_DIR = os.path.join(HARNESS_ROOT, "driver")


class DriverError(Exception):
    pass


class ElectronSession:
    """One fresh app profile. Never points at the owner's profile: userData must live under `scratch_root`."""

    def __init__(self, electron_bin, app_index, scratch_root, session_id, allowed_port, stderr_path=None, startup_timeout=60):
        if int(allowed_port) in FORBIDDEN_PORTS:
            raise DriverError("allowed port is forbidden")
        if int(allowed_port) not in HARNESS_PORTS:
            raise DriverError("allowed port %s is not a harness port %s" % (allowed_port, HARNESS_PORTS))
        for p in (electron_bin, app_index, scratch_root):
            if not os.path.isabs(p):
                raise DriverError("absolute paths required: %s" % p)
        real_profile = os.path.expanduser("~/Library/Application Support/soloroleplayer-m-electron")
        self.dir = os.path.join(scratch_root, "session_" + session_id)
        if os.path.realpath(self.dir).startswith(os.path.realpath(real_profile)):
            raise DriverError("refusing to use the owner's real profile folder")
        self.user_data = os.path.join(self.dir, "userdata")
        os.makedirs(self.user_data, exist_ok=True)
        self.cfg_path = os.path.join(self.dir, "session.json")
        with open(self.cfg_path, "w") as f:
            json.dump({"appIndex": app_index, "preload": os.path.join(DRIVER_DIR, "preload_stub.cjs"),
                       "pageDriver": os.path.join(DRIVER_DIR, "page_driver.js"), "userData": self.user_data,
                       "allowedHost": "127.0.0.1", "allowedPort": int(allowed_port), "forbiddenPorts": list(FORBIDDEN_PORTS), "harnessPorts": list(HARNESS_PORTS)}, f)
        self.electron_bin = electron_bin
        self.stderr_path = stderr_path
        self.startup_timeout = startup_timeout
        self.proc = None
        self._q = queue.Queue()
        self._id = 0
        self.events = []
        self.stray_stdout = []
        self.ready = None
        self._desc = []

    def start(self):
        errf = open(self.stderr_path, "ab") if self.stderr_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([self.electron_bin, os.path.join(DRIVER_DIR, "electron_main.cjs"), self.cfg_path],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1,
                                     start_new_session=True)
        self.pgid = os.getpgid(self.proc.pid)
        threading.Thread(target=self._reader, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise DriverError("electron exited during startup (code %s)" % self.proc.returncode)
                continue
            if msg.get("event") == "ready":
                self.ready = msg
                self._desc = descendants(self.proc.pid)
                return msg
            if msg.get("event") == "fatal":
                raise DriverError("electron fatal: %s" % msg.get("error"))
            self.events.append(msg)
        raise DriverError("electron not ready within %ss" % self.startup_timeout)

    def _reader(self):
        for line in self.proc.stdout:
            if line.startswith(SENTINEL):
                try:
                    self._q.put(json.loads(line[len(SENTINEL):]))
                except Exception:
                    pass
            else:
                self.stray_stdout.append(line[:300])
        self._q.put({"event": "eof"})

    def request(self, cmd, args=None, timeout=900):
        if self.proc is None or self.proc.poll() is not None:
            raise DriverError("electron is not running")
        self._id += 1
        rid = self._id
        self.proc.stdin.write(json.dumps({"id": rid, "cmd": cmd, "args": args or {}}) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while True:
            left = timeout - (time.time() - t0)
            if left <= 0:
                raise DriverError("timeout waiting for %s" % cmd)
            try:
                msg = self._q.get(timeout=min(1.0, left))
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise DriverError("electron died while running %s (code %s)" % (cmd, self.proc.returncode))
                continue
            if msg.get("event") == "eof":
                raise DriverError("electron closed its output during %s" % cmd)
            if msg.get("event"):
                self.events.append(msg)
                continue
            if msg.get("id") == rid:
                if not msg.get("ok"):
                    raise DriverError("%s failed: %s" % (cmd, msg.get("error")))
                return msg["result"]

    def close(self, delete_scratch=True):
        """Shut Electron down (whole process group) and verify nothing is left. Returns the verification dict."""
        v = {"ok": True}
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.request("shutdown", timeout=10)
            except Exception:
                pass
            t0 = time.time()
            while self.proc.poll() is None and time.time() - t0 < 5:
                time.sleep(0.1)
            import signal
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.pgid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                time.sleep(0.3)
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            time.sleep(0.2)
            v = verify_process_tree_gone(self.proc.pid, self.pgid, self._desc)
            v["ok"] = not v["pid_alive"] and not v["group_alive"] and not v["descendants_alive"]
        if delete_scratch:
            real = os.path.realpath(self.dir)
            if "_scratch" in real and os.path.isdir(real):  # only ever delete our own throwaway folder
                shutil.rmtree(real, ignore_errors=True)
        return v
