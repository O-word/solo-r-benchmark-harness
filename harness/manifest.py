"""MANIFEST.sha256: a hash of every file in a run folder (tamper-evident, append-only).

A finished run folder is never edited. Files added later (hand labels, agreement reports, re-judging) get their own
MANIFEST_addendum_<timestamp>.sha256; the original MANIFEST.sha256 is never rewritten.
Filesystem noise (._* AppleDouble sidecars on exFAT, .DS_Store) is excluded.
"""
import glob
import os
import time

from .util import is_ignored_file, sha256_file

MANIFEST = "MANIFEST.sha256"


def _files(run_dir, exclude=()):
    out = []
    for dp, dn, fn in os.walk(run_dir):
        dn.sort()
        for n in sorted(fn):
            if is_ignored_file(n):
                continue
            rel = os.path.relpath(os.path.join(dp, n), run_dir).replace(os.sep, "/")
            if rel == MANIFEST or rel.startswith("MANIFEST_addendum_") or rel in exclude:
                continue
            out.append(rel)
    return sorted(out)


def write_manifest(run_dir):
    lines = ["%s  %s" % (sha256_file(os.path.join(run_dir, rel)), rel) for rel in _files(run_dir)]
    with open(os.path.join(run_dir, MANIFEST), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def write_addendum(run_dir, new_files):
    """Hash files added after the run finished into a separate addendum manifest."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(run_dir, "MANIFEST_addendum_%s.sha256" % ts)
    lines = ["%s  %s" % (sha256_file(os.path.join(run_dir, rel)), rel) for rel in sorted(new_files)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def verify_manifest(run_dir):
    """Return dict(ok, checked, missing, changed, unlisted)."""
    listed = {}
    manifests = [os.path.join(run_dir, MANIFEST)] + sorted(glob.glob(os.path.join(run_dir, "MANIFEST_addendum_*.sha256")))
    if not os.path.exists(manifests[0]):
        return {"ok": False, "error": "no MANIFEST.sha256", "checked": 0, "missing": [], "changed": [], "unlisted": []}
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    h, rel = line.split("  ", 1)
                    listed[rel] = h
    missing, changed = [], []
    for rel, h in listed.items():
        p = os.path.join(run_dir, rel)
        if not os.path.exists(p):
            missing.append(rel)
        elif sha256_file(p) != h:
            changed.append(rel)
    unlisted = [r for r in _files(run_dir) if r not in listed]
    for r in list(unlisted):
        if r.startswith("MANIFEST"):
            unlisted.remove(r)
    return {"ok": not (missing or changed or unlisted), "checked": len(listed), "missing": missing, "changed": changed, "unlisted": unlisted}
