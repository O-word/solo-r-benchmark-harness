"""Read-side helpers for run folders: which attempts of which sessions count.

A session can be attempted more than once in the same run folder (a crash, a stop, or a resume re-runs it from the start on a
fresh instance). Every attempt's records stay in the files (append-only). Aggregates use, per session, the LAST attempt whose
summary says 'complete' (or 'incomplete', i.e. ended early on a bot failure but ran honestly on a fresh instance).
Attempts that were interrupted or crashed, and attempts superseded by a later one, are kept but excluded.
"""
import csv
import os

from .util import read_jsonl

COUNTED_STATUSES = ("complete", "incomplete")


def load_sessions(run_dir):
    return [r for r in read_jsonl(os.path.join(run_dir, "sessions.jsonl")) if r.get("event") == "session"]


def valid_attempts(sessions):
    """{session_id: attempt} for the attempt whose records count."""
    best = {}
    for s in sessions:
        if s.get("status") in COUNTED_STATUSES:
            a = int(s.get("attempt", 1))
            if s["session_id"] not in best or a > best[s["session_id"]]:
                best[s["session_id"]] = a
    return best


def valid_session_summaries(sessions):
    v = valid_attempts(sessions)
    return [s for s in sessions if s.get("status") in COUNTED_STATUSES and int(s.get("attempt", 1)) == v.get(s["session_id"])]


def load_rows(run_dir, only_valid=True):
    with open(os.path.join(run_dir, "scores.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not only_valid:
        return rows
    v = valid_attempts(load_sessions(run_dir))
    return [r for r in rows if v.get(r["session_id"]) == int(r.get("attempt") or 1)]


def uncounted_sessions(run_dir, planned_ids):
    v = valid_attempts(load_sessions(run_dir))
    return [sid for sid in planned_ids if sid not in v]
