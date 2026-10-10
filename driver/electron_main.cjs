// Headless Electron driver for the Solo R Benchmark Harness.
//
// Usage (always with ABSOLUTE paths; the Python side builds them):
//   Electron electron_main.cjs /abs/path/session_config.json
//
// session_config.json:
//   { appIndex, preload, pageDriver, userData, allowedHost, allowedPort, forbiddenPorts, harnessPorts }
//
// Protocol on stdin/stdout: one JSON object per line.
//   in : {"id": 1, "cmd": "setup"|"turn"|"app_command"|"state"|"blocked"|"probe_blocked"|"shutdown", "args": {...}}
//   out: "@@HF@@" + {"id": 1, "ok": true, "result": ...} or {"id":1,"ok":false,"error":"..."}
//   Unsolicited: {"event":"ready"|"log", ...}
//
// Isolation guarantees implemented here:
//   * userData (and every other Electron path) lives inside the run folder, never the real profile.
//   * every network request is cancelled unless it targets http://<allowedHost>:<allowedPort>/ or
//     is a local file/data/blob URL; requests to forbidden ports are refused even if allowed by mistake.
//     allowedPort must be a harness port (1236 local, 1237 stub, 1238 SSH tunnel to an external server); see netfilter.cjs.
const { app, BrowserWindow, session } = require("electron");
app.disableHardwareAcceleration(); // headless harness window needs no GPU; keeps the GPU free for the model and the owner's desktop
const fs = require("fs");
const path = require("path");
const readline = require("readline");

const cfgPath = process.argv[2];
if (!cfgPath || !path.isAbsolute(cfgPath)) {
  process.stdout.write("@@HF@@" + JSON.stringify({ event: "fatal", error: "session config path must be absolute" }) + "\n");
  process.exit(2);
}
const cfg = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
for (const k of ["appIndex", "preload", "pageDriver", "userData"]) {
  if (!path.isAbsolute(cfg[k])) {
    process.stdout.write("@@HF@@" + JSON.stringify({ event: "fatal", error: k + " must be an absolute path" }) + "\n");
    process.exit(2);
  }
}
const { makeFilter } = require("./netfilter.cjs");
const FORBIDDEN = new Set((cfg.forbiddenPorts || [1234, 1235]).map(Number));
const ALLOWED_PORT = Number(cfg.allowedPort);
let FILTER;
try {
  FILTER = makeFilter({ allowedHost: cfg.allowedHost || "127.0.0.1", allowedPort: ALLOWED_PORT, forbiddenPorts: Array.from(FORBIDDEN), harnessPorts: cfg.harnessPorts });
} catch (e) {
  process.stdout.write("@@HF@@" + JSON.stringify({ event: "fatal", error: String(e && e.message || e) }) + "\n");
  process.exit(2);
}

// Everything Electron writes goes inside the throwaway folder.
fs.mkdirSync(cfg.userData, { recursive: true });
for (const name of ["userData", "sessionData", "cache", "logs", "crashDumps", "temp"]) {
  try { app.setPath(name, path.join(cfg.userData, name)); fs.mkdirSync(path.join(cfg.userData, name), { recursive: true }); } catch (e) {}
}
app.commandLine.appendSwitch("no-sandbox");
app.commandLine.appendSwitch("disable-background-timer-throttling");
app.commandLine.appendSwitch("disable-renderer-backgrounding");
app.commandLine.appendSwitch("disable-gpu");
app.disableHardwareAcceleration();

const blocked = [];
function out(obj) { process.stdout.write("@@HF@@" + JSON.stringify(obj) + "\n"); }


app.whenReady().then(async () => {
  try { if (process.platform === "darwin") app.setActivationPolicy("accessory"); } catch (e) {}
  const ses = session.defaultSession;
  ses.webRequest.onBeforeRequest({ urls: ["<all_urls>"] }, (details, callback) => {
    const ok = FILTER.allowed(details.url);
    if (!ok) {
      blocked.push({ t: Date.now(), method: details.method, url: details.url.replace(/\?.*$/, "").slice(0, 200) });
      return callback({ cancel: true });
    }
    callback({});
  });
  ses.setPermissionRequestHandler((wc, perm, cb) => cb(false));

  const win = new BrowserWindow({
    show: false, width: 1400, height: 900,
    webPreferences: { preload: cfg.preload, contextIsolation: true, nodeIntegration: false, sandbox: false, backgroundThrottling: false }
  });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e, url) => { if (!/^file:/i.test(url)) e.preventDefault(); });
  win.webContents.on("console-message", (e, level, msg) => { if (level >= 2) out({ event: "log", level, msg: String(msg).slice(0, 300) }); });
  win.webContents.on("render-process-gone", (e, d) => out({ event: "log", level: 3, msg: "render-process-gone " + JSON.stringify(d) }));

  try {
    await win.loadFile(cfg.appIndex);
    const driverSrc = fs.readFileSync(cfg.pageDriver, "utf8");
    await win.webContents.executeJavaScript(driverSrc, true);
  } catch (err) {
    out({ event: "fatal", error: String(err && err.stack || err) });
    return app.exit(3);
  }
  out({ event: "ready", versions: { electron: process.versions.electron, chrome: process.versions.chrome, node: process.versions.node }, userData: app.getPath("userData") });

  async function run(code) { return await win.webContents.executeJavaScript(code, true); }
  async function handle(msg) {
    const args = JSON.stringify(msg.args === undefined ? {} : msg.args);
    switch (msg.cmd) {
      case "setup": return await run(`window.__HF.setup(${args})`);
      case "turn": return await run(`window.__HF.turn(${args})`);
      case "app_command": return await run(`window.__HF.appCommand(${args})`);
      case "state": return await run(`window.__HF.stateSnapshot()`);
      case "blocked": return { blocked };
      case "probe_blocked": {
        // Ask the page to fetch a port that is NOT allowed (never 1234/1235) and report how it failed.
        const p = Number(msg.args && msg.args.port);
        if (FORBIDDEN.has(p) || p === ALLOWED_PORT) throw new Error("probe port must be neither allowed nor forbidden");
        const before = blocked.length;
        const verdict = await run(`(async()=>{try{await window.__HF.nativeFetch("http://127.0.0.1:${p}/probe");return "reached"}catch(e){return String(e&&e.message||e)}})()`);
        return { verdict, newlyBlocked: blocked.length - before };
      }
      case "shutdown": return { bye: true };
      default: throw new Error("unknown cmd " + msg.cmd);
    }
  }

  const rl = readline.createInterface({ input: process.stdin });
  let chain = Promise.resolve();
  rl.on("line", (line) => {
    line = line.trim();
    if (!line) return;
    let msg;
    try { msg = JSON.parse(line); } catch (e) { return out({ id: null, ok: false, error: "bad json" }); }
    chain = chain.then(async () => {
      try {
        const result = await handle(msg);
        out({ id: msg.id, ok: true, result });
        if (msg.cmd === "shutdown") setTimeout(() => app.exit(0), 50);
      } catch (e) {
        out({ id: msg.id, ok: false, error: String(e && e.message || e).slice(0, 2000) });
      }
    });
  });
  rl.on("close", () => app.exit(0));
});

app.on("window-all-closed", () => {});
