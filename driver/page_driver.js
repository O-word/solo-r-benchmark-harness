// Injected into the REAL app page (index.html) by electron_main.cjs. Nothing here replaces the
// app's briefing builder, model call or reply guards: a turn is driven by calling the app's own
// handleInput() with a say / : / @emit command, exactly like typing it into the app.
//
// What this file adds around the app:
//   * a fetch wrapper that (a) refuses any URL outside the one allowed origin, (b) records every
//     request and response (the exact briefing), (c) optionally injects seed / temperature /
//     max_tokens / model into the request body (recorded as "injected").
//   * a switch for the reply guards: the app's hasXxx() predicates and the paragraph-count check
//     are wrapped. With guards off they report "no problem" (so no rewrite call happens) but the
//     real answer is still evaluated and logged as guard_events ("would have fired").
(() => {
  if (window.__HF) return "already";
  const HF = window.__HF = {
    version: "1",
    calls: [],
    guardEvents: [],
    guardsOn: true,
    inTurn: false,
    allowedOrigin: null,
    inject: { seed: null, temperature: null, top_p: null, max_tokens: null, model: null },
    modelName: null,
    nativeFetch: window.fetch.bind(window),
    pageBlocked: []
  };

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  // ---------- fetch wrapper ----------
  window.fetch = async function (url, opts) {
    const u = String(url);
    if (HF.allowedOrigin && !(u === HF.allowedOrigin || u.startsWith(HF.allowedOrigin + "/"))) {
      HF.pageBlocked.push(u.replace(/\?.*$/, "").slice(0, 200));
      throw new TypeError("harness: request blocked (not the allowed origin): " + u.replace(/\?.*$/, "").slice(0, 120));
    }
    const o = Object.assign({}, opts || {});
    let bodyObj = null;
    let injected = null;
    const isChat = /\/chat(\/completions)?$/.test(u.split("?")[0]);
    if (isChat && typeof o.body === "string") {
      try { bodyObj = JSON.parse(o.body); } catch (e) { bodyObj = null; }
      if (bodyObj && typeof bodyObj === "object") {
        injected = {};
        const sys = String(bodyObj.system_prompt || (bodyObj.messages && bodyObj.messages[0] && bodyObj.messages[0].content) || "");
        const isScribe = sys.startsWith("You are the SCRIBE");
        if (HF.inject.model) { if (bodyObj.model !== HF.inject.model) injected.model = HF.inject.model; bodyObj.model = HF.inject.model; }
        if (HF.inject.seed != null) { bodyObj.seed = HF.inject.seed; injected.seed = HF.inject.seed; }
        if (!isScribe && HF.inject.temperature != null && bodyObj.temperature == null) { bodyObj.temperature = HF.inject.temperature; injected.temperature = HF.inject.temperature; }
        if (!isScribe && HF.inject.top_p != null && bodyObj.top_p == null) { bodyObj.top_p = HF.inject.top_p; injected.top_p = HF.inject.top_p; }
        if (!isScribe && HF.inject.max_tokens != null && bodyObj.max_tokens == null) { bodyObj.max_tokens = HF.inject.max_tokens; injected.max_tokens = HF.inject.max_tokens; }
        o.body = JSON.stringify(bodyObj);
      }
    }
    const rec = { n: HF.calls.length, url: u, path: u.replace(/^https?:\/\/[^/]+/, "").split("?")[0], t0: Date.now(), request: bodyObj, injected, status: null, response_text: null, ms: null, error: null };
    HF.calls.push(rec);
    try {
      const res = await HF.nativeFetch(u, o);
      const text = await res.text();
      rec.status = res.status;
      rec.response_text = text;
      rec.ms = Date.now() - rec.t0;
      return new Response(text, { status: res.status, statusText: res.statusText, headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" } });
    } catch (e) {
      rec.error = String(e && e.message || e).slice(0, 300);
      rec.ms = Date.now() - rec.t0;
      throw e;
    }
  };

  // ---------- guard switch ----------
  const GUARD_FUNCS = ["hasPlayerOwnershipViolation", "hasNarrativeFirstPersonDrift", "hasScreenplayFormattingDrift",
    "hasNarrationPreferenceDrift", "hasUnpairedPunctuationDrift", "hasIdentityDrift"];
  const origFns = {};
  let guardsInstalled = false;
  function installGuardWrappers() {
    if (guardsInstalled) return;
    for (const name of GUARD_FUNCS) {
      if (typeof window[name] !== "function") throw new Error("app function missing: " + name);
      origFns[name] = window[name];
      window[name] = function (...args) {
        const real = origFns[name].apply(this, args);
        if (HF.inTurn) HF.guardEvents.push({ name, real: !!real, applied: HF.guardsOn ? !!real : false });
        return HF.guardsOn ? real : false;
      };
    }
    // Freshness guard (added to the app 2026-10-08): switched off with the other guards. Optional so older app snapshots still work.
    if (typeof window.freshnessEnabled === "function") {
      origFns.freshnessEnabled = window.freshnessEnabled;
      window.freshnessEnabled = function (...args) {
        const real = origFns.freshnessEnabled.apply(this, args);
        if (HF.inTurn) HF.guardEvents.push({ name: "freshnessEnabled", real: !!real, applied: HF.guardsOn ? !!real : false });
        return HF.guardsOn ? real : false;
      };
    }
    if (typeof window.countParagraphs !== "function") throw new Error("app function missing: countParagraphs");
    origFns.countParagraphs = window.countParagraphs;
    window.countParagraphs = function (text) {
      const real = origFns.countParagraphs.call(this, text);
      // Only the paragraph-count GUARD inside runHoloQuickRequest is switched off. countParagraphs is also called from the app's
      // sanitizer (normalizeCompanionParagraphs), which is not a guard and must keep working, so look at the direct caller frame.
      if (HF.inTurn && !HF.guardsOn) {
        const frames = (new Error().stack || "").split("\n");
        const caller = frames[2] || "";
        if (/runHoloQuickRequest/.test(caller)) {
          HF.guardEvents.push({ name: "countParagraphs", real, applied: false });
          return 9999; // paragraph-count guard disabled: never "too short"
        }
      }
      return real;
    };
    guardsInstalled = true;
  }
  HF.origFns = origFns;

  // ---------- helpers ----------
  function classify(rec) {
    const r = rec.request || {};
    const sys = String(r.system_prompt || (r.messages && r.messages[0] && r.messages[0].content) || "");
    const usr = String(r.input || (r.messages && r.messages[1] && r.messages[1].content) || "");
    if (/\/models$/.test(rec.path)) return "model_list";
    if (!/\/chat(\/completions)?$/.test(rec.path)) return "other";
    if (sys.startsWith("You are the SCRIBE")) return "scribe";
    if (/^\/api\/v1\/chat$/.test(rec.path) && [404, 405, 501].includes(rec.status)) return "transport_probe";
    if (/\bRewrite\b/.test(sys)) return "rewrite";
    return "reply";
  }
  function briefingOf(rec) {
    const r = rec.request || {};
    const system = String(r.system_prompt || (r.messages && r.messages[0] && r.messages[0].content) || "");
    const user = String(r.input || (r.messages && r.messages[1] && r.messages[1].content) || "");
    return { system, user };
  }
  function textOf(rec) {
    try { return window.extractModelText(JSON.parse(rec.response_text)); } catch (e) { return ""; }
  }
  function sceneSnapshot() {
    const ss = (state.sceneTracker && state.sceneTracker.sceneState) || {};
    const pick = ["playerWardrobeCue", "companionWardrobeCue", "locationAnchor", "stagingAnchor", "playerHolding", "companionHolding",
      "timeOfDay", "playerSubLocation", "companionSubLocation", "objectOfInterest", "whoHasIt", "mood", "currentConflict"];
    const o = {};
    for (const k of pick) o[k] = ss[k] || "";
    o.scribeFacts = (ss.scribeFacts || []).slice();
    o.activeThreads = (ss.activeThreads || []).slice();
    try { o.outfitsLine = currentOutfitsLine(); } catch (e) { o.outfitsLine = ""; }
    o.currentRoom = state.currentRoom;
    return o;
  }
  HF.stateSnapshot = function () {
    return {
      mode: state.mode,
      endpoint: state.modelSettings && state.modelSettings.companion && state.modelSettings.companion.endpoint,
      model: state.modelSettings && state.modelSettings.companion && state.modelSettings.companion.model,
      companionName: state.companion && state.companion.name,
      playerName: state.playerProfile && state.playerProfile.displayName,
      narration: narrationPreferenceList(),
      level: currentRPStyleLevel(),
      requiredParagraphs: requiredRPParagraphs(),
      personaInjection: state.companion && state.companion.personaInjection,
      scene: sceneSnapshot(),
      pageBlocked: HF.pageBlocked.slice()
    };
  };

  // ---------- setup: build the companion + settings programmatically ----------
  HF.setup = async function (a) {
    installGuardWrappers();
    HF.allowedOrigin = a.allowedOrigin;
    HF.inject = Object.assign({ seed: null, temperature: null, top_p: null, max_tokens: null, model: null }, a.inject || {});
    HF.guardsOn = a.guardsOn !== false;
    // The exact model name for this configuration (e.g. "base" / "lora" on one external server). Set in the companion AND helper
    // (scribe) settings; re-asserted before every turn (the app can re-detect a model from /v1/models) and forced into every chat
    // request body by the fetch wrapper (inject.model), so the name that goes over the wire is the configured one.
    HF.modelName = a.model;

    const ms = state.modelSettings;
    for (const key of ["companion", "helper"]) {
      ms[key].endpoint = a.endpoint;
      ms[key].model = a.model;
    }
    if (a.temperature != null) ms.companion.temperature = a.temperature;
    if (a.maxTokens != null) ms.companion.maxTokens = a.maxTokens;
    state.aiRuntime.endpoint = a.endpoint;
    state.aiRuntime.activeModel = a.model;
    state.aiRuntime.status = "connected";
    state.debugRuntime = false;

    // Companion: the app's own normalizer, fed persona fields, Hard Rules, Important Notes.
    state.companion = normalizeCompanionTemplate(Object.assign({}, defaultCompanionProfile(), a.companion || {}));

    // Player profile, narration perspective + tense, Roleplay Style Level.
    state.playerProfile = Object.assign({}, state.playerProfile, {
      displayName: a.player.name,
      fullName: a.player.name,
      gender: a.player.gender || "Unspecified",
      allowedNarrationTenses: buildAllowedNarrationTenses(a.perspective, a.tense),
      rpStyleLevel: String(a.level)
    });
    state.worldSetup = Object.assign({}, state.worldSetup || {}, { rpStyleLevel: String(a.level) });

    // Preference Bible (player preferences).
    state.preferenceBible = normalizePreferenceBibleList(a.preferences || []);

    // Scene memory.
    state.stasisScribe = Object.assign({ enabled: true, every: 3, window: 8, maxTokens: 560 }, a.scribe || {});
    state.sceneTracker = { active: true, startedAt: new Date().toISOString(), kind: "fresh", seedSummary: "", rollingSummary: [], sceneState: sceneStateDefaults() };
    state.mode = "holo";
    state.holoRunning = true;
    if (a.startRoom && state.rooms && state.rooms[a.startRoom]) state.currentRoom = a.startRoom;
    // Optional: lengthen the app's own reply timeout (it is 90 s for an unknown model name such as default_model; a long
    // Level 3 reply on a 16 GB Mac can exceed that). Recorded in config.json; null keeps the app's value.
    if (a.appTimeoutMs) { HF.appTimeoutMs = a.appTimeoutMs; window.localModelTimeoutMs = function () { return HF.appTimeoutMs; }; }
    // Scenario rooms: same fields a player edits with @customize here.
    for (const [rid, r] of Object.entries(a.rooms || {})) {
      if (state.rooms && state.rooms[rid]) {
        if (r.name) state.rooms[rid].name = r.name;
        if (r.description) state.rooms[rid].description = r.description;
      }
    }
    return HF.stateSnapshot();
  };

  // A non-pose app action done by the benchmark script (room change). Only /go <direction> is allowed.
  HF.appCommand = async function (a) {
    if (!/^\/go (north|south|east|west|up|down|n|s|e|w|u|d)$/i.test(String(a.command || ""))) throw new Error("harness: only '/go <direction>' app commands are allowed");
    const logBefore = (els.log.textContent || "").length;
    await handleInput(a.command);
    return { command: a.command, scene: sceneSnapshot(), log_delta: (els.log.textContent || "").slice(logBefore).slice(0, 1500) };
  };

  // ---------- one turn ----------
  const POSE_RE = /^(say |s |:|@emit |@e )/i;
  HF.turn = async function (a) {
    if (!POSE_RE.test(String(a.command || ""))) throw new Error("harness: player poses must start with say, s, : or @emit");
    HF.calls = [];
    HF.guardEvents = [];
    HF.guardsOn = a.guardsOn !== false;
    HF.inject.seed = a.seed != null ? a.seed : HF.inject.seed;
    const logBefore = (els.log.textContent || "").length;
    const scribeLogBefore = ((state.sceneTracker && state.sceneTracker.scribeLog) || []).length;
    const t0 = Date.now();
    let final = null, err = null;
    const modelBefore = { companion: state.modelSettings.companion.model, helper: state.modelSettings.helper.model, active: state.aiRuntime.activeModel };
    if (HF.modelName) {
      state.modelSettings.companion.model = HF.modelName;
      state.modelSettings.helper.model = HF.modelName;
      state.aiRuntime.activeModel = HF.modelName;
    }
    HF.inTurn = true;
    const timeoutMs = a.timeoutMs || 600000;
    try {
      final = await Promise.race([
        handleInput(a.command),
        sleep(timeoutMs).then(() => { throw new Error("turn timeout after " + timeoutMs + " ms"); })
      ]);
    } catch (e) { err = String(e && e.message || e).slice(0, 500); }
    const tModel = Date.now() - t0;
    HF.inTurn = false;
    let scribeWaitedMs = 0;
    if (a.waitScribe) {
      const tw = Date.now();
      while (state.sceneTracker && state.sceneTracker.scribeRunning && Date.now() - tw < (a.scribeTimeoutMs || 120000)) await sleep(25);
      scribeWaitedMs = Date.now() - tw;
    }
    const calls = HF.calls.map(rec => {
      const kind = classify(rec);
      return { n: rec.n, kind, path: rec.path, status: rec.status, ms: rec.ms, error: rec.error, injected: rec.injected,
        request: rec.request, response_text: rec.response_text, text: kind === "reply" || kind === "rewrite" ? textOf(rec) : undefined };
    });
    const replyCalls = calls.filter(c => c.kind === "reply");
    const first = replyCalls[0] || null;
    const raw_text = first ? first.text : null;
    let sanitized = null;
    try { sanitized = raw_text != null ? sanitizeAIText(raw_text) : null; } catch (e) { sanitized = raw_text; }
    const scribeLog = ((state.sceneTracker && state.sceneTracker.scribeLog) || []).slice(scribeLogBefore);
    return {
      command: a.command,
      final_text: typeof final === "string" ? final : null,
      error: err,
      ms_model: tModel,
      scribe_waited_ms: scribeWaitedMs,
      briefing: first ? briefingOf(first.request ? first : { request: null }) : null,
      briefing_url: first ? first.path : null,
      raw_text,
      sanitized_text: sanitized,
      rewrite_count: calls.filter(c => c.kind === "rewrite").length,
      calls,
      guard_events: HF.guardEvents.slice(),
      guards_on: HF.guardsOn,
      model_setting: { configured: HF.modelName, before_turn: modelBefore, after_turn: { companion: state.modelSettings.companion.model, helper: state.modelSettings.helper.model, active: state.aiRuntime.activeModel } },
      scribe_log: scribeLog,
      log_delta: (els.log.textContent || "").slice(logBefore).slice(0, 6000),
      required_paragraphs: requiredRPParagraphs(),
      level: currentRPStyleLevel(),
      narration: narrationPreferenceList(),
      scene: sceneSnapshot(),
      page_blocked: HF.pageBlocked.slice()
    };
  };

  return "installed";
})();
