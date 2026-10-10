// Harness preload. REPLACES the app's own preload.cjs on purpose.
//
// The app's preload.cjs exposes readLegacyShards(), which reads *.shard files from the
// real user's ~/Downloads and ~/Library/Application Support/com.soloroleplayer.* folders.
// The benchmark must never touch those (isolation rule 1), so the harness exposes a
// stub with the same name that returns nothing and reports a non-electron runtime so the
// app's legacy-import code takes its "unsupported" branch and never calls it.
const { contextBridge } = require("electron");

contextBridge.exposeInMainWorld("soloDesktop", {
  platform: process.platform,
  runtime: "eval-harness-stub",
  readLegacyShards: () => []
});
