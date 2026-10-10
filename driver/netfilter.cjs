// Network allowlist used by electron_main.cjs (pure Node, no Electron API, so it can be unit-tested).
//
// A request is allowed only if it is a local file/data/blob/devtools URL, or it starts with exactly
//   http://<allowedHost>:<allowedPort>/
// where allowedPort is ONE of the harness ports (1236 local model, 1237 stub, 1238 SSH-tunnel to an external server).
// Ports 1234 and 1235 (the owner's play servers) are refused even if some config tried to allow them.
"use strict";

const DEFAULT_FORBIDDEN = [1234, 1235];
const DEFAULT_HARNESS_PORTS = [1236, 1237, 1238];

function makeFilter(opts) {
  const forbidden = new Set((opts.forbiddenPorts || DEFAULT_FORBIDDEN).map(Number));
  const harnessPorts = new Set((opts.harnessPorts || DEFAULT_HARNESS_PORTS).map(Number));
  const port = Number(opts.allowedPort);
  const host = opts.allowedHost || "127.0.0.1";
  if (forbidden.has(port)) throw new Error("allowed port is a forbidden port");
  if (!harnessPorts.has(port)) throw new Error("allowed port " + port + " is not a harness port (" + Array.from(harnessPorts).join("/") + ")");
  if (host !== "127.0.0.1") throw new Error("allowed host must be 127.0.0.1");
  const prefix = `http://${host}:${port}/`;
  function allowed(url) {
    url = String(url);
    let ok = false;
    if (/^(file|data|blob|devtools|about|chrome|chrome-extension):/i.test(url)) ok = true;
    else if (url.startsWith(prefix) || url === prefix.slice(0, -1)) ok = true;
    try {
      const u = new URL(url);
      if (/^https?:$/.test(u.protocol) && forbidden.has(Number(u.port))) ok = false;
    } catch (e) {}
    return ok;
  }
  return { allowed, prefix, port, host };
}

module.exports = { makeFilter, DEFAULT_FORBIDDEN, DEFAULT_HARNESS_PORTS };
