"""Start / health-check / stop the harness's OWN model server, then prove it is gone.

Ports: the real server uses 1236, the stub uses 1237. Ports 1234 and 1235 belong to the owner's play servers
and are refused everywhere: this module never connects to them, never reads their logs, and never kills a
process it did not start.

The real launch command is a PLACEHOLDER (configs/real_template.json) to be filled in from Cute LM's actual
launch arguments later. Any token beginning with REPLACE_ME makes start() refuse; nothing is guessed.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .util import EXTERNAL_PORTS, FORBIDDEN_PORTS, HARNESS_PORTS, HARNESS_ROOT, REAL_PORT, STUB_PORT, now_iso

ALLOWED_PORTS = (REAL_PORT, STUB_PORT)       # ports a LOCAL server (one the harness spawns) may use


class ServerError(Exception):
    pass


class ExternalServerDown(ServerError):
    """The external (tunnelled) server did not answer, or does not serve the expected model name. The run pauses (resumable)."""
    pass


def check_port_allowed(port, allowed=None):
    """Default: ports a harness-spawned server may use (1236, 1237). Pass allowed=EXTERNAL_PORTS or HARNESS_PORTS for the others."""
    allowed = ALLOWED_PORTS if allowed is None else allowed
    if int(port) in FORBIDDEN_PORTS:
        raise ServerError("port %s is the owner's play-server port; the harness must never use it" % port)
    if int(port) not in allowed:
        raise ServerError("port %s is not an allowed harness port (allowed: %s)" % (port, tuple(allowed)))
    return int(port)


def port_in_use(port):
    """True if something is listening on 127.0.0.1:port (checked by trying to bind; only ever touches `port`)."""
    check_port_allowed(port, HARNESS_PORTS)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", int(port)))
        return False
    except OSError:
        return True
    finally:
        s.close()


def listeners_on(port):
    """PIDs listening on the port, via lsof restricted to that single port."""
    check_port_allowed(port, HARNESS_PORTS)
    try:
        out = subprocess.run(["lsof", "-nP", "-iTCP:%d" % int(port), "-sTCP:LISTEN", "-t"], capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def _children(pid):
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def descendants(pid):
    seen, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c in _children(p):
            if c not in seen:
                seen.append(c)
                todo.append(c)
    return seen


def stray_processes(port):
    """Processes whose command line looks like a model server bound to OUR port (e.g. mlx_lm.server ... --port 1236).
    Reads the process table only; matches on '--port <port>' so the owner's servers (other ports) can never match."""
    check_port_allowed(port, HARNESS_PORTS)
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    needles = ("--port %d" % port, "--port=%d" % port)
    me = os.getpid()
    res = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "ps -axo" in line:
            continue
        pid_s, _, cmd = line.partition(" ")
        if pid_s.isdigit() and int(pid_s) != me and any(n in cmd for n in needles) and ("mlx" in cmd or "server" in cmd):
            res.append({"pid": int(pid_s), "command": cmd[:200]})
    return res


def required_paths(command):
    """Executable and the file/dir arguments that must exist before a real server is started (stat only; nothing is loaded)."""
    toks = [str(t) for t in command]
    paths = [toks[0]] if toks else []
    for flag in ("--model", "--adapter-path", "--draft-model"):
        if flag in toks and toks.index(flag) + 1 < len(toks):
            paths.append(toks[toks.index(flag) + 1])
    return paths


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # zombies count as gone
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except Exception:
        return True


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def verify_process_tree_gone(pid, pgid, descendant_pids):
    alive_desc = [p for p in descendant_pids if pid_alive(p)]
    return {"pid_alive": pid_alive(pid), "group_alive": group_alive(pgid), "descendants_alive": alive_desc}


def http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


class ServerManager:
    def __init__(self, command, port, kind="real", cwd=None, env=None, log_path=None, health_urls=None,
                 health_timeout=180.0, stop_grace=15.0):
        self.port = check_port_allowed(port)
        self.kind = kind
        self.command = [str(c).replace("{port}", str(self.port)) for c in command]
        self.cwd = cwd
        self.env = dict(os.environ, **(env or {}))
        self.log_path = log_path
        if health_urls:
            self.health_urls = health_urls
        elif kind == "real":
            self.health_urls = ["http://127.0.0.1:%d/v1/models" % self.port]      # readiness = /v1/models answers
        else:
            self.health_urls = ["http://127.0.0.1:%d/health" % self.port, "http://127.0.0.1:%d/v1/models" % self.port]
        self.health_timeout = health_timeout
        self.stop_grace = stop_grace
        self.proc = None
        self.info = None
        self._desc = []
        self._logf = None

    # -- start ---------------------------------------------------------------------------------
    def preflight(self):
        if not self.command:
            raise ServerError("empty server command")
        for tok in self.command:
            if tok.startswith("REPLACE_ME"):
                raise ServerError("server command still contains the placeholder %r; fill it in from Cute LM's launch arguments (do not guess)" % tok)
        for tok in self.command:
            if tok in ("1234", "1235") or tok.endswith(":1234") or tok.endswith(":1235"):
                raise ServerError("server command mentions a forbidden port: %r" % tok)
        if "--port" in self.command and self.command[self.command.index("--port") + 1] != str(self.port):
            raise ServerError("command --port does not match the harness port %d" % self.port)
        if self.kind == "real":
            for p in required_paths(self.command):
                if not os.path.exists(p):
                    raise ServerError("required path does not exist (drive not mounted?): %s" % p)
        if port_in_use(self.port):
            raise ServerError("port %d is already in use; refusing to start (and refusing to kill unknown processes)" % self.port)
        if stray_processes(self.port):
            raise ServerError("a process already looks like a model server on port %d: %s" % (self.port, stray_processes(self.port)))

    def start(self):
        self.preflight()
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=self._logf or subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
        self.info = {"kind": self.kind, "pid": self.proc.pid, "pgid": os.getpgid(self.proc.pid), "port": self.port,
                     "argv": list(self.command), "start_time": now_iso(), "start_monotonic": time.monotonic()}
        try:
            self.wait_healthy()
        except Exception:
            self.stop()
            raise
        self._desc = descendants(self.proc.pid)
        return dict(self.info)

    def wait_healthy(self):
        t0 = time.time()
        while time.time() - t0 < self.health_timeout:
            if self.proc.poll() is not None:
                raise ServerError("server exited during startup with code %s (see %s)" % (self.proc.returncode, self.log_path))
            if any(http_ok(u) for u in self.health_urls):
                self.info["healthy_after_s"] = round(time.time() - t0, 2)
                return
            time.sleep(0.2)
        raise ServerError("server not healthy within %.0f s" % self.health_timeout)

    def age_s(self):
        return round(time.monotonic() - self.info["start_monotonic"], 1) if self.info else None

    # -- stop + verify ---------------------------------------------------------------------------
    def stop(self):
        """Terminate the whole process group, then verify the process, descendants and port are gone."""
        if not self.proc:
            return {"stopped": True, "note": "never started"}
        pid, pgid = self.proc.pid, self.info["pgid"] if self.info else self.proc.pid
        self._desc = sorted(set(self._desc) | set(descendants(pid)))
        for sig, wait in ((signal.SIGTERM, self.stop_grace), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                # macOS raises EPERM when the group has already exited and only a zombie is left; verify_gone() below
                # still proves the process tree and port are gone, so a real failure is not hidden.
                break
            t0 = time.time()
            while time.time() - t0 < wait:
                if self.proc.poll() is not None and not group_alive(pgid):
                    break
                time.sleep(0.1)
            if self.proc.poll() is not None and not group_alive(pgid):
                break
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        if self._logf:
            try:
                self._logf.close()
            except Exception:
                pass
        v = self.verify_gone()
        self.info = dict(self.info or {}, stopped_at=now_iso())
        return v

    def verify_gone(self):
        res = verify_process_tree_gone(self.proc.pid, self.info["pgid"], self._desc)
        # the OS can take a moment to release the socket
        for _ in range(50):
            if not port_in_use(self.port):
                break
            time.sleep(0.1)
        res["port"] = self.port
        res["port_free"] = not port_in_use(self.port)
        res["port_listeners"] = listeners_on(self.port)
        res["stray_model_processes"] = stray_processes(self.port)
        res["ok"] = (not res["pid_alive"] and not res["group_alive"] and not res["descendants_alive"]
                     and res["port_free"] and not res["port_listeners"] and not res["stray_model_processes"])
        return res


def harness_descendants():
    """Live (non-zombie) descendants of THIS python process: proof the harness leaves nothing running locally."""
    return [p for p in descendants(os.getpid()) if pid_alive(p)]


def harness_electron_processes():
    """Processes running THIS harness's Electron driver script (reads the process table only)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    key = os.path.join(HARNESS_ROOT, "driver", "electron_main.cjs")
    return [l.strip()[:160] for l in out.splitlines() if key in l and "ps -axo" not in l]


class ExternalServerDown(ServerError):
    """The external (tunnelled) server did not answer, or does not serve the expected model name. The run pauses (resumable)."""
    pass


def check_port_allowed(port, allowed=None):
    """Default: ports a harness-spawned server may use (1236, 1237). Pass allowed=EXTERNAL_PORTS or HARNESS_PORTS for the others."""
    allowed = ALLOWED_PORTS if allowed is None else allowed
    if int(port) in FORBIDDEN_PORTS:
        raise ServerError("port %s is the owner's play-server port; the harness must never use it" % port)
    if int(port) not in allowed:
        raise ServerError("port %s is not an allowed harness port (allowed: %s)" % (port, tuple(allowed)))
    return int(port)


def port_in_use(port):
    """True if something is listening on 127.0.0.1:port (checked by trying to bind; only ever touches `port`)."""
    check_port_allowed(port, HARNESS_PORTS)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", int(port)))
        return False
    except OSError:
        return True
    finally:
        s.close()


def listeners_on(port):
    """PIDs listening on the port, via lsof restricted to that single port."""
    check_port_allowed(port, HARNESS_PORTS)
    try:
        out = subprocess.run(["lsof", "-nP", "-iTCP:%d" % int(port), "-sTCP:LISTEN", "-t"], capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def _children(pid):
    try:
        out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=10).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def descendants(pid):
    seen, todo = [], [pid]
    while todo:
        p = todo.pop()
        for c in _children(p):
            if c not in seen:
                seen.append(c)
                todo.append(c)
    return seen


def stray_processes(port):
    """Processes whose command line looks like a model server bound to OUR port (e.g. mlx_lm.server ... --port 1236).
    Reads the process table only; matches on '--port <port>' so the owner's servers (other ports) can never match."""
    check_port_allowed(port, HARNESS_PORTS)
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    needles = ("--port %d" % port, "--port=%d" % port)
    me = os.getpid()
    res = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "ps -axo" in line:
            continue
        pid_s, _, cmd = line.partition(" ")
        if pid_s.isdigit() and int(pid_s) != me and any(n in cmd for n in needles) and ("mlx" in cmd or "server" in cmd):
            res.append({"pid": int(pid_s), "command": cmd[:200]})
    return res


def required_paths(command):
    """Executable and the file/dir arguments that must exist before a real server is started (stat only; nothing is loaded)."""
    toks = [str(t) for t in command]
    paths = [toks[0]] if toks else []
    for flag in ("--model", "--adapter-path", "--draft-model"):
        if flag in toks and toks.index(flag) + 1 < len(toks):
            paths.append(toks[toks.index(flag) + 1])
    return paths


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # zombies count as gone
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return bool(out) and not out.startswith("Z")
    except Exception:
        return True


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def verify_process_tree_gone(pid, pgid, descendant_pids):
    alive_desc = [p for p in descendant_pids if pid_alive(p)]
    return {"pid_alive": pid_alive(pid), "group_alive": group_alive(pgid), "descendants_alive": alive_desc}


def http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


class ServerManager:
    def __init__(self, command, port, kind="real", cwd=None, env=None, log_path=None, health_urls=None,
                 health_timeout=180.0, stop_grace=15.0):
        self.port = check_port_allowed(port)
        self.kind = kind
        self.command = [str(c).replace("{port}", str(self.port)) for c in command]
        self.cwd = cwd
        self.env = dict(os.environ, **(env or {}))
        self.log_path = log_path
        if health_urls:
            self.health_urls = health_urls
        elif kind == "real":
            self.health_urls = ["http://127.0.0.1:%d/v1/models" % self.port]      # readiness = /v1/models answers
        else:
            self.health_urls = ["http://127.0.0.1:%d/health" % self.port, "http://127.0.0.1:%d/v1/models" % self.port]
        self.health_timeout = health_timeout
        self.stop_grace = stop_grace
        self.proc = None
        self.info = None
        self._desc = []
        self._logf = None

    # -- start ---------------------------------------------------------------------------------
    def preflight(self):
        if not self.command:
            raise ServerError("empty server command")
        for tok in self.command:
            if tok.startswith("REPLACE_ME"):
                raise ServerError("server command still contains the placeholder %r; fill it in from Cute LM's launch arguments (do not guess)" % tok)
        for tok in self.command:
            if tok in ("1234", "1235") or tok.endswith(":1234") or tok.endswith(":1235"):
                raise ServerError("server command mentions a forbidden port: %r" % tok)
        if "--port" in self.command and self.command[self.command.index("--port") + 1] != str(self.port):
            raise ServerError("command --port does not match the harness port %d" % self.port)
        if self.kind == "real":
            for p in required_paths(self.command):
                if not os.path.exists(p):
                    raise ServerError("required path does not exist (drive not mounted?): %s" % p)
        if port_in_use(self.port):
            raise ServerError("port %d is already in use; refusing to start (and refusing to kill unknown processes)" % self.port)
        if stray_processes(self.port):
            raise ServerError("a process already looks like a model server on port %d: %s" % (self.port, stray_processes(self.port)))

    def start(self):
        self.preflight()
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._logf = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.command, cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                                     stdout=self._logf or subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True)
        self.info = {"kind": self.kind, "pid": self.proc.pid, "pgid": os.getpgid(self.proc.pid), "port": self.port,
                     "argv": list(self.command), "start_time": now_iso(), "start_monotonic": time.monotonic()}
        try:
            self.wait_healthy()
        except Exception:
            self.stop()
            raise
        self._desc = descendants(self.proc.pid)
        return dict(self.info)

    def wait_healthy(self):
        t0 = time.time()
        while time.time() - t0 < self.health_timeout:
            if self.proc.poll() is not None:
                raise ServerError("server exited during startup with code %s (see %s)" % (self.proc.returncode, self.log_path))
            if any(http_ok(u) for u in self.health_urls):
                self.info["healthy_after_s"] = round(time.time() - t0, 2)
                return
            time.sleep(0.2)
        raise ServerError("server not healthy within %.0f s" % self.health_timeout)

    def age_s(self):
        return round(time.monotonic() - self.info["start_monotonic"], 1) if self.info else None

    # -- stop + verify ---------------------------------------------------------------------------
    def stop(self):
        """Terminate the whole process group, then verify the process, descendants and port are gone."""
        if not self.proc:
            return {"stopped": True, "note": "never started"}
        pid, pgid = self.proc.pid, self.info["pgid"] if self.info else self.proc.pid
        self._desc = sorted(set(self._desc) | set(descendants(pid)))
        for sig, wait in ((signal.SIGTERM, self.stop_grace), (signal.SIGKILL, 5.0)):
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                # macOS raises EPERM when the group has already exited and only a zombie is left; verify_gone() below
                # still proves the process tree and port are gone, so a real failure is not hidden.
                break
            t0 = time.time()
            while time.time() - t0 < wait:
                if self.proc.poll() is not None and not group_alive(pgid):
                    break
                time.sleep(0.1)
            if self.proc.poll() is not None and not group_alive(pgid):
                break
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        if self._logf:
            try:
                self._logf.close()
            except Exception:
                pass
        v = self.verify_gone()
        self.info = dict(self.info or {}, stopped_at=now_iso())
        return v

    def verify_gone(self):
        res = verify_process_tree_gone(self.proc.pid, self.info["pgid"], self._desc)
        # the OS can take a moment to release the socket
        for _ in range(50):
            if not port_in_use(self.port):
                break
            time.sleep(0.1)
        res["port"] = self.port
        res["port_free"] = not port_in_use(self.port)
        res["port_listeners"] = listeners_on(self.port)
        res["stray_model_processes"] = stray_processes(self.port)
        res["ok"] = (not res["pid_alive"] and not res["group_alive"] and not res["descendants_alive"]
                     and res["port_free"] and not res["port_listeners"] and not res["stray_model_processes"])
        return res


def harness_descendants():
    """Live (non-zombie) descendants of THIS python process: proof the harness leaves nothing running locally."""
    return [p for p in descendants(os.getpid()) if pid_alive(p)]


def harness_electron_processes():
    """Processes whose command line is this harness's Electron driver (reads the process table only)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    key = "driver/electron_main.cjs"
    return [l.strip()[:160] for l in out.splitlines() if key in l and "ps -axo" not in l and "SoloRoleplayer" in l.replace("\\", "/") or (key in l and "eval_harness" in l and "ps -axo" not in l)]


class ExternalServer:
    """An already-running OpenAI-compatible server reached through a local port (the SSH tunnel listens on 127.0.0.1:1238).

    The harness NEVER starts, stops, signals or inspects the remote process, and never touches the tunnel. All it does is
    GET http://127.0.0.1:<port>/v1/models (readiness + the served model names). It exposes the same small interface as
    ServerManager (start / stop / info / age_s) so the runner can treat both alike.
    """
    kind = "external"
    proc = None

    def __init__(self, port, required_models=None, health_timeout=30.0, poll_interval=1.0, request_timeout=5.0):
        self.port = check_port_allowed(port, EXTERNAL_PORTS)
        self.base_url = "http://127.0.0.1:%d" % self.port
        self.required_models = [m for m in (required_models or []) if m]
        self.health_timeout = float(health_timeout)
        self.poll_interval = float(poll_interval)
        self.request_timeout = float(request_timeout)
        self.info = None
        # direct connection to 127.0.0.1 only: never through an environment proxy
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def models(self):
        """List of served model ids; raises on any failure (connection refused, reset, bad status, not JSON)."""
        with self._opener.open(self.base_url + "/v1/models", timeout=self.request_timeout) as r:
            if not (200 <= r.status < 300):
                raise ServerError("HTTP %s from /v1/models" % r.status)
            data = json.loads(r.read().decode("utf-8", "replace"))
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [str(m.get("id", "")).strip() for m in data["data"] if isinstance(m, dict) and str(m.get("id", "")).strip()]
        raise ServerError("/v1/models answered but not in OpenAI list format")

    def probe(self):
        """One non-raising readiness probe: {reachable, models, error}. Sends no chat request."""
        try:
            return {"reachable": True, "models": self.models(), "error": None, "port": self.port, "base_url": self.base_url}
        except Exception as e:
            return {"reachable": False, "models": [], "error": "%s: %s" % (type(e).__name__, str(e)[:160]), "port": self.port, "base_url": self.base_url}

    def missing_models(self, served):
        return [m for m in self.required_models if m not in served]

    def wait_healthy(self, timeout=None):
        t0 = time.time()
        timeout = self.health_timeout if timeout is None else timeout
        last = None
        while True:
            pr = self.probe()
            if pr["reachable"]:
                miss = self.missing_models(pr["models"])
                if not miss:
                    return pr
                last = "server answers but does not serve %s (serves: %s)" % (miss, pr["models"])
            else:
                last = pr["error"]
            if time.time() - t0 >= timeout:
                raise ExternalServerDown("external server %s not ready within %.0f s: %s" % (self.base_url, timeout, last))
            time.sleep(min(self.poll_interval, max(0.05, timeout - (time.time() - t0))))

    def start(self):
        """'Connect': wait until /v1/models answers and lists every required model name. Starts nothing."""
        t0 = time.time()
        pr = self.wait_healthy()
        self.info = {"kind": "external", "pid": None, "pgid": None, "port": self.port, "argv": ["external", self.base_url], "start_time": now_iso(),
                     "start_monotonic": time.monotonic(), "healthy_after_s": round(time.time() - t0, 2), "base_url": self.base_url,
                     "served_models": pr["models"], "required_models": list(self.required_models),
                     "note": "external server: the harness did not start it and will not stop it; start_time/age are of this harness connection, not of the remote process"}
        return dict(self.info)

    def check(self, grace_s=0.0):
        """Mid-run health check: {ok, models, error}. With grace_s > 0 it keeps retrying for that long before reporting failure."""
        t0 = time.time()
        while True:
            pr = self.probe()
            if pr["reachable"] and not self.missing_models(pr["models"]):
                return {"ok": True, "models": pr["models"], "error": None}
            err = pr["error"] if not pr["reachable"] else "missing model(s) %s; serves %s" % (self.missing_models(pr["models"]), pr["models"])
            if time.time() - t0 >= grace_s:
                return {"ok": False, "models": pr["models"], "error": err}
            time.sleep(min(2.0, max(0.05, grace_s - (time.time() - t0))))

    def age_s(self):
        return round(time.monotonic() - self.info["start_monotonic"], 1) if self.info else None

    def stop(self):
        """No-op by design: the remote process and the tunnel are not ours to stop."""
        return {"ok": True, "external": True, "stopped": False, "remote_process_untouched": True, "tunnel_untouched": True,
                "note": "external server: nothing to stop or verify; the harness never signals the remote process or the tunnel"}


def make_stub_command(mode="mixed", port=STUB_PORT, extra=None, log=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "stub_server.py"), "--port", "{port}", "--mode", mode]
    if log:
        cmd += ["--log", log]
    return cmd + list(extra or [])
