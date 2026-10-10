"""Minimal Gemini API client (stdlib only) shared by the Gemini player bot and the Gemini judge.

KEY HANDLING (design rule): the key lives in ~/.solo_bench_gemini.env as GEMINI_API_KEY=... (or in the
GEMINI_API_KEY environment variable). It is read only at call time, sent only in the x-goog-api-key header,
never put in a URL, never logged, never written to any output, and every error string is scrubbed of it.
`key_status()` reports whether a key is available WITHOUT reading its value (file existence / env presence only).

Untested against the live API until a key exists. Network access goes through `transport`, which tests replace.
"""
import json
import os
import random
import time
import urllib.error
import urllib.request

KEY_FILE = os.path.expanduser("~/.solo_bench_gemini.env")
ENV_VAR = "GEMINI_API_KEY"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models/"


class GeminiError(Exception):
    """Raised after retries are exhausted or on a non-retryable error. Message is already key-scrubbed."""

    def __init__(self, msg, status=None, retryable=False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


def key_status(key_file=KEY_FILE):
    """('env'|'file'|'missing', detail) without reading the key value."""
    if os.environ.get(ENV_VAR):
        return "env", "environment variable %s is set" % ENV_VAR
    if os.path.isfile(key_file):
        return "file", "key file exists at %s" % key_file
    return "missing", "no %s in the environment and no key file at %s" % (ENV_VAR, key_file)


def _read_key(key_file=KEY_FILE):
    v = os.environ.get(ENV_VAR)
    if v:
        return v.strip()
    if os.path.isfile(key_file):
        with open(key_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(ENV_VAR + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _default_transport(url, headers, body, timeout):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()


class GeminiClient:
    def __init__(self, model, key_file=KEY_FILE, timeout=90, max_retries=5, base_delay=2.0, max_delay=60.0,
                 min_interval=0.0, transport=None, sleep=time.sleep):
        self.model = model
        self.key_file = key_file
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.min_interval = min_interval
        self.transport = transport or _default_transport
        self.sleep = sleep
        self._last = 0.0
        self.stats = {"calls": 0, "retries": 0, "rate_limited": 0, "errors": 0}

    def available(self):
        return key_status(self.key_file)[0] != "missing"

    def _scrub(self, s, key):
        s = str(s)
        return s.replace(key, "[REDACTED]") if key else s

    def generate(self, system, user, temperature=0.7, seed=None, json_mode=False, max_output_tokens=1200, thinking_budget=None, response_schema=None):
        """Return dict(text, usage, attempts, finish_reason, raw_status). Raises GeminiError on failure."""
        key = _read_key(self.key_file)
        if not key:
            raise GeminiError("no Gemini key available (set %s or create %s)" % (ENV_VAR, self.key_file), retryable=False)
        gen = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        if seed is not None:
            gen["seed"] = int(seed)
        if json_mode:
            gen["responseMimeType"] = "application/json"
        if response_schema is not None:
            gen["responseSchema"] = response_schema   # structured output: the API guarantees valid JSON of this shape (added 2026-10-09 after malformed-JSON aborts)
        if thinking_budget is not None:
            # Gemini 2.5 "thinking" tokens count against maxOutputTokens: without a budget they can eat the whole allowance and leave truncated JSON (found 2026-10-09).
            gen["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen,
        }).encode("utf-8")
        url = API_ROOT + self.model + ":generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        last_err = None
        for attempt in range(1, self.max_retries + 2):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.stats["calls"] += 1
            try:
                status, hdrs, raw = self.transport(url, headers, body, self.timeout)
            except Exception as e:  # network error, timeout
                last_err = GeminiError("network error: %s" % self._scrub(type(e).__name__ + ": " + str(e), key), retryable=True)
                status, hdrs, raw = None, {}, b""
            if status == 200:
                try:
                    data = json.loads(raw.decode("utf-8"))
                    cand = (data.get("candidates") or [{}])[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    text = "".join(p.get("text", "") for p in parts)
                    if not text.strip():
                        last_err = GeminiError("empty response (finish_reason=%s)" % cand.get("finishReason"), status=200, retryable=True)
                    else:
                        return {"text": text, "usage": data.get("usageMetadata"), "attempts": attempt,
                                "finish_reason": cand.get("finishReason"), "raw_status": 200}
                except Exception as e:
                    last_err = GeminiError("unparseable response: %s" % self._scrub(e, key), status=200, retryable=True)
            elif status is not None:
                snippet = self._scrub(raw[:300].decode("utf-8", "replace"), key)
                if status == 429:
                    self.stats["rate_limited"] += 1
                if status in (429, 500, 502, 503, 504):
                    last_err = GeminiError("HTTP %s: %s" % (status, snippet), status=status, retryable=True)
                else:
                    self.stats["errors"] += 1
                    raise GeminiError("HTTP %s: %s" % (status, snippet), status=status, retryable=False)
            if attempt > self.max_retries or last_err is None or not last_err.retryable:
                break
            self.stats["retries"] += 1
            retry_after = None
            try:
                ra = {k.lower(): v for k, v in (hdrs or {}).items()}.get("retry-after")
                retry_after = float(ra) if ra else None
            except Exception:
                retry_after = None
            delay = retry_after if retry_after is not None else min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
            self.sleep(delay + random.uniform(0, 0.25 * delay))
        self.stats["errors"] += 1
        raise last_err or GeminiError("unknown failure")
