// Test helper: load the pinned app (harness preload, no network, throwaway profile) and evaluate a JS file in the page.
const { app, BrowserWindow, session } = require("electron");
const fs = require("fs"), path = require("path");
const [indexHtml, preload, testFile, userData] = process.argv.slice(2);
for (const n of ["userData", "sessionData", "cache", "logs", "crashDumps"]) { try { app.setPath(n, path.join(userData, n)); fs.mkdirSync(path.join(userData, n), { recursive: true }); } catch (e) {} }
app.commandLine.appendSwitch("no-sandbox");
app.whenReady().then(async () => {
  session.defaultSession.webRequest.onBeforeRequest({ urls: ["<all_urls>"] }, (d, cb) => cb({ cancel: !/^(file|data|blob|devtools):/.test(d.url) }));
  const win = new BrowserWindow({ show: false, webPreferences: { preload, contextIsolation: true, sandbox: false } });
  await win.loadFile(indexHtml);
  const out = await win.webContents.executeJavaScript(fs.readFileSync(testFile, "utf8"), true);
  console.log("RESULT " + JSON.stringify(out));
  app.exit(0);
});
