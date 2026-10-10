"""Small shared helpers (stdlib only)."""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_PORTS = (1234, 1235)       # the owner's play servers: never contacted, never read
REAL_PORT = 1236                     # the harness's own real model-server port
STUB_PORT = 1237                     # the stub server used for dry runs and tests
TUNNEL_PORT = 1238                   # local end of the SSH tunnel to an EXTERNAL (remote, already-running) OpenAI-compatible server
EXTERNAL_PORTS = (STUB_PORT, TUNNEL_PORT)   # ports an "external" server may use: the tunnel, or the stub (tests / dry run only)
HARNESS_PORTS = (REAL_PORT, STUB_PORT, TUNNEL_PORT)   # the only ports the harness (and Electron's network filter) may ever talk to
IGNORED_NAMES = {".DS_Store"}


def is_ignored_file(name):
    """macOS/exFAT AppleDouble sidecars (._*) and .DS_Store are filesystem noise, not run content."""
    return name.startswith("._") or name in IGNORED_NAMES


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_hash(root, rel_paths=None):
    """Hash of sorted '<sha256>  <relpath>' lines for every real file under root."""
    lines = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for n in sorted(fn):
            if is_ignored_file(n):
                continue
            full = os.path.join(dp, n)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel_paths is not None and rel not in rel_paths:
                continue
            lines.append("%s  %s" % (sha256_file(full), rel))
    return sha256_text("\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1]))), lines


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


class JsonlAppender:
    """Append-only JSON-lines writer, flushed after every record so a crash never loses recorded turns."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def write(self, obj):
        self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def seed_from(*parts):
    """Deterministic 31-bit seed from arbitrary parts."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def strip_pose_markers(text):
    """Pose text as the player typed it: %r / %t markers become whitespace."""
    return re.sub(r"%[rRtT]", " ", text)


def monotonic_ms():
    return int(time.monotonic() * 1000)
