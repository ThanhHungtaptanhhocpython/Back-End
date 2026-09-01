import assert from "node:assert/strict";
import test from "node:test";

import {
  YT_API_SRC,
  YT_PLAYER_STATE,
  describeYouTubeError,
  loadYouTubeIframeApi,
  resetYouTubeApiLoader,
  runYouTubePlayerEffect,
  startYouTubePlayback,
} from "../src/services/youtubePlayer.js";

/* ------------------------------------------------------------------ *
 * Test doubles
 * ------------------------------------------------------------------ */

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Stand-in for `YT.Player`. Records every instance and exposes event pokes. */
class FakePlayer {
  static instances = [];
  static reset() {
    FakePlayer.instances = [];
  }

  constructor(host, config) {
    this.host = host;
    this.config = config || {};
    this.events = this.config.events || {};
    this.destroyed = false;
    this._state = YT_PLAYER_STATE.UNSTARTED;
    this._time = 0;
    this._loaded = 0;
    FakePlayer.instances.push(this);
  }

  destroy() {
    this.destroyed = true;
  }
  getPlayerState() {
    return this._state;
  }
  getCurrentTime() {
    return this._time;
  }
  getVideoLoadedFraction() {
    return this._loaded;
  }
  playVideo() {
    this._state = YT_PLAYER_STATE.PLAYING;
    this.emitState(YT_PLAYER_STATE.PLAYING);
  }
  pauseVideo() {
    this._state = YT_PLAYER_STATE.PAUSED;
  }
  seekTo(seconds) {
    this._time = seconds;
  }

  emitReady(target = this) {
    this.events.onReady?.({ target });
  }
  emitState(data, target = this) {
    this.events.onStateChange?.({ data, target });
  }
  emitError(data, target = this) {
    this.events.onError?.({ data, target });
  }
  emitBlocked(target = this) {
    this.events.onAutoplayBlocked?.({ target });
  }
}

/** Minimal stand-in for the React-owned outer container. */
function fakeContainer({ connected = true } = {}) {
  const ownerDocument = {
    createElement: (tag) => ({
      tagName: String(tag).toUpperCase(),
      _attrs: {},
      setAttribute(key, value) {
        this._attrs[key] = value;
      },
      isConnected: false,
    }),
  };
  return {
    ownerDocument,
    children: [],
    replaceChildren(...nodes) {
      this.children.forEach((node) => {
        node.isConnected = false;
      });
      this.children = nodes;
      nodes.forEach((node) => {
        node.isConnected = true;
      });
    },
    appendChild(node) {
      this.children.push(node);
      node.isConnected = connected;
      return node;
    },
    querySelector() {
      return null;
    },
  };
}

/** Minimal DOM for exercising the API loader. */
function makeFakeDom() {
  const scripts = [];
  const makeEl = () => {
    const el = {
      _listeners: {},
      src: "",
      async: false,
      setAttribute() {},
      remove() {
        const i = scripts.indexOf(el);
        if (i >= 0) scripts.splice(i, 1);
      },
      addEventListener(type, fn) {
        (el._listeners[type] ||= []).push(fn);
      },
      dispatchEvent(type) {
        (el._listeners[type] || []).slice().forEach((fn) => fn({ type }));
      },
    };
    return el;
  };
  const doc = {
    querySelector(selector) {
      const match = /^script\[src="(.+)"\]$/.exec(selector);
      if (!match) return null;
      return scripts.find((s) => s.src === match[1]) || null;
    },
    createElement() {
      return makeEl();
    },
    head: {
      appendChild(el) {
        scripts.push(el);
      },
    },
  };
  const win = { location: { origin: "http://localhost:5173" } };
  return { win, doc, scripts };
}

/* ------------------------------------------------------------------ *
 * describeYouTubeError - numeric code mapping
 * ------------------------------------------------------------------ */

test("describeYouTubeError maps every documented player error code", () => {
  assert.equal(describeYouTubeError(2).code, 2);
  assert.match(describeYouTubeError(2).message, /invalid|parameter/i);
  assert.match(describeYouTubeError(5).message, /HTML5|player failed/i);
  assert.match(describeYouTubeError(100).message, /removed|private/i);
  assert.match(describeYouTubeError(101).message, /embed/i);
  assert.match(describeYouTubeError(150).message, /embed/i);
  assert.match(describeYouTubeError(153).message, /Referer|origin/i);

  const unknown = describeYouTubeError(999);
  assert.equal(unknown.code, 999);
  assert.match(unknown.message, /999/);

  const garbage = describeYouTubeError("weird");
  assert.equal(garbage.code, 0);
  assert.match(garbage.message, /weird/);
});

/* ------------------------------------------------------------------ *
 * startYouTubePlayback - construction
 * ------------------------------------------------------------------ */

test("startYouTubePlayback builds the player with cross-origin playerVars", async () => {
  let built;
  const stop = startYouTubePlayback({
    host: { isConnected: true },
    videoId: "mjy8h-iT-ms",
    startSeconds: 12.9,
    origin: "http://localhost:5173",
    loadApi: async () => ({}),
    createPlayer: (_YT, host, config) => {
      built = { host, config };
      return { destroy() {} };
    },
  });
  await flush();

  assert.equal(built.config.videoId, "mjy8h-iT-ms");
  assert.deepEqual(built.config.playerVars, {
    start: 12,
    autoplay: 1,
    rel: 0,
    playsinline: 1,
    enablejsapi: 1,
    origin: "http://localhost:5173",
  });
  stop();
});

test("startYouTubePlayback honours autoplay: false", async () => {
  let vars;
  const stop = startYouTubePlayback({
    host: { isConnected: true },
    videoId: "abc",
    autoplay: false,
    origin: "http://localhost:5173",
    loadApi: async () => ({}),
    createPlayer: (_YT, _host, config) => {
      vars = config.playerVars;
      return { destroy() {} };
    },
  });
  await flush();
  assert.equal(vars.autoplay, 0);
  stop();
});

test("startYouTubePlayback reports an error when the host is detached before load", async () => {
  const errors = [];
  const stop = startYouTubePlayback({
    host: { isConnected: false },
    videoId: "abc",
    loadApi: async () => ({ Player: FakePlayer }),
    onError: (info) => errors.push(info),
  });
  await flush();
  assert.equal(errors.length, 1);
  assert.equal(errors[0].code, 0);
  assert.match(errors[0].message, /detached/i);
  stop();
});

test("startYouTubePlayback routes a loader rejection to onError", async () => {
  const errors = [];
  const stop = startYouTubePlayback({
    host: { isConnected: true },
    videoId: "abc",
    loadApi: async () => {
      throw new Error("boom");
    },
    onError: (info) => errors.push(info),
  });
  await flush();
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /boom/);
  stop();
});

/* ------------------------------------------------------------------ *
 * runYouTubePlayerEffect - lifecycle contract
 * ------------------------------------------------------------------ */

test("a fresh inner host is created inside the React-owned container each run", async () => {
  FakePlayer.reset();
  const container = fakeContainer();
  const cleanup = runYouTubePlayerEffect({
    container,
    videoId: "abc",
    playerRef: { current: null },
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();

  assert.equal(container.children.length, 1);
  const host = container.children[0];
  assert.equal(host._attrs["data-yt-host"], "abc");
  assert.equal(FakePlayer.instances[0].host, host);
  cleanup();
});

test("happy path: onReady publishes event.target into playerRef", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const seen = [];
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    startSeconds: 30,
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
    onReady: (info) => seen.push(info),
  });
  await flush();

  const player = FakePlayer.instances[0];
  player._state = YT_PLAYER_STATE.PLAYING;
  player.emitReady();

  assert.equal(playerRef.current, player);
  assert.equal(seen.length, 1);
  assert.equal(seen[0].player, player);
  assert.equal(seen[0].state, YT_PLAYER_STATE.PLAYING);
  cleanup();
  assert.equal(playerRef.current, null);
  assert.equal(player.destroyed, true);
});

test("event.target - not the constructor return - is the authoritative player", async () => {
  const playerRef = { current: null };
  const constructed = { tag: "constructed", destroy() {} };
  const authoritative = {
    tag: "authoritative",
    getPlayerState: () => YT_PLAYER_STATE.PLAYING,
    getCurrentTime: () => 0,
    getVideoLoadedFraction: () => 0,
  };
  let events;
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    playerRef,
    loadApi: async () => ({}),
    createPlayer: (_YT, _host, config) => {
      events = config.events;
      return constructed;
    },
  });
  await flush();

  events.onReady({ target: authoritative });
  assert.equal(playerRef.current, authoritative);
  assert.notEqual(playerRef.current, constructed);
  cleanup();
});

test("delayed API resolution after cleanup never builds a player", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const api = deferred();
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    playerRef,
    loadApi: () => api.promise,
    onReady: () => assert.fail("onReady must not fire after cleanup"),
  });

  cleanup(); // teardown before the API promise settles
  api.resolve({ Player: FakePlayer });
  await flush();

  assert.equal(FakePlayer.instances.length, 0);
  assert.equal(playerRef.current, null);
});

test("StrictMode setup -> cleanup -> setup leaves exactly one live player", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const container = fakeContainer();

  const apiA = deferred();
  const cleanupA = runYouTubePlayerEffect({
    container,
    videoId: "abc",
    playerRef,
    loadApi: () => apiA.promise,
  });
  cleanupA();

  const apiB = deferred();
  const cleanupB = runYouTubePlayerEffect({
    container,
    videoId: "abc",
    playerRef,
    loadApi: () => apiB.promise,
  });

  apiA.resolve({ Player: FakePlayer });
  apiB.resolve({ Player: FakePlayer });
  await flush();

  assert.equal(FakePlayer.instances.length, 1);
  const live = FakePlayer.instances[0];
  live.emitReady();
  assert.equal(playerRef.current, live);

  cleanupB();
  assert.equal(playerRef.current, null);
  assert.equal(live.destroyed, true);
});

test("a late onReady from a destroyed player is ignored", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
    onReady: () => assert.fail("stale onReady must not fire"),
    onStateChange: () => assert.fail("stale onStateChange must not fire"),
  });
  await flush();

  const player = FakePlayer.instances[0];
  cleanup();
  player.emitReady();
  player.emitState(YT_PLAYER_STATE.PLAYING);

  assert.equal(playerRef.current, null);
});

test("hide then re-show builds a fresh player and reuses the empty container", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const container = fakeContainer();

  const cleanup1 = runYouTubePlayerEffect({
    container,
    videoId: "abc",
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();
  FakePlayer.instances[0].emitReady();
  cleanup1(); // hide

  assert.equal(playerRef.current, null);
  assert.equal(container.children.length, 0);

  const cleanup2 = runYouTubePlayerEffect({
    container,
    videoId: "abc",
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();
  const reshown = FakePlayer.instances[1];
  reshown.emitReady();

  assert.equal(playerRef.current, reshown);
  assert.notEqual(reshown, FakePlayer.instances[0]);
  assert.equal(FakePlayer.instances[0].destroyed, true);
  cleanup2();
});

test("a videoId / startSeconds change tears down the old run and rebuilds with new vars", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const container = fakeContainer();

  const cleanup1 = runYouTubePlayerEffect({
    container,
    videoId: "first",
    startSeconds: 5,
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();
  FakePlayer.instances[0].emitReady();
  cleanup1();

  const cleanup2 = runYouTubePlayerEffect({
    container,
    videoId: "second",
    startSeconds: 42,
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();

  const next = FakePlayer.instances[1];
  assert.equal(next.config.videoId, "second");
  assert.equal(next.config.playerVars.start, 42);
  assert.equal(next.config.playerVars.enablejsapi, 1);
  assert.equal(FakePlayer.instances[0].destroyed, true);
  cleanup2();
});

test("a stale cleanup cannot erase a newer player ref", async () => {
  FakePlayer.reset();
  const playerRef = { current: null };
  const container = fakeContainer();

  const cleanupA = runYouTubePlayerEffect({
    container,
    videoId: "v1",
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();
  const playerA = FakePlayer.instances[0];
  playerA.emitReady();
  assert.equal(playerRef.current, playerA);

  const cleanupB = runYouTubePlayerEffect({
    container,
    videoId: "v2",
    playerRef,
    loadApi: async () => ({ Player: FakePlayer }),
  });
  await flush();
  const playerB = FakePlayer.instances[1];
  playerB.emitReady();
  assert.equal(playerRef.current, playerB);

  // A's cleanup runs late (hide/show race). It must not null the newer ref.
  cleanupA();
  assert.equal(playerRef.current, playerB);
  assert.equal(playerA.destroyed, true);

  cleanupB();
  assert.equal(playerRef.current, null);
});

test("autoplay-blocked is delivered separately from onError", async () => {
  FakePlayer.reset();
  const events = [];
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    playerRef: { current: null },
    loadApi: async () => ({ Player: FakePlayer }),
    onError: (info) => events.push(["error", info]),
    onAutoplayBlocked: () => events.push(["blocked"]),
    onStateChange: (info) => events.push(["state", info.state]),
  });
  await flush();

  const player = FakePlayer.instances[0];
  player.emitReady();
  player.emitBlocked();
  player.emitError(150);

  assert.deepEqual(
    events.filter((e) => e[0] === "blocked"),
    [["blocked"]],
  );
  const err = events.find((e) => e[0] === "error");
  assert.equal(err[1].code, 150);
  assert.match(err[1].message, /embed/i);
  cleanup();
});

test("numeric error codes from onStateChange/onError reach the effect consumer mapped", async () => {
  FakePlayer.reset();
  const errors = [];
  const cleanup = runYouTubePlayerEffect({
    container: fakeContainer(),
    videoId: "abc",
    playerRef: { current: null },
    loadApi: async () => ({ Player: FakePlayer }),
    onError: (info) => errors.push(info),
  });
  await flush();
  const player = FakePlayer.instances[0];
  player.emitReady();
  player.emitError(101);
  player.emitError(2);

  assert.deepEqual(
    errors.map((e) => e.code),
    [101, 2],
  );
  assert.match(errors[0].message, /embed/i);
  assert.match(errors[1].message, /invalid|parameter/i);
  cleanup();
});

/* ------------------------------------------------------------------ *
 * loadYouTubeIframeApi - loader hardening
 * ------------------------------------------------------------------ */

test("loadYouTubeIframeApi hands one shared promise to concurrent callers", async () => {
  resetYouTubeApiLoader();
  const { win, doc, scripts } = makeFakeDom();

  const p1 = loadYouTubeIframeApi({ win, doc });
  const p2 = loadYouTubeIframeApi({ win, doc });
  assert.equal(p1, p2);
  assert.equal(scripts.length, 1);

  win.YT = { Player: function Player() {} };
  win.onYouTubeIframeAPIReady();

  const [a, b] = await Promise.all([p1, p2]);
  assert.equal(a, win.YT);
  assert.equal(b, win.YT);
  resetYouTubeApiLoader();
});

test("loadYouTubeIframeApi chains, and does not clobber, an existing ready handler", async () => {
  resetYouTubeApiLoader();
  const { win, doc } = makeFakeDom();
  let prevCalls = 0;
  win.onYouTubeIframeAPIReady = () => {
    prevCalls += 1;
  };

  const p = loadYouTubeIframeApi({ win, doc });
  win.YT = { Player: function Player() {} };
  win.onYouTubeIframeAPIReady();

  await p;
  assert.equal(prevCalls, 1);
  resetYouTubeApiLoader();
});

test("loadYouTubeIframeApi reuses a <script> another consumer already added", async () => {
  resetYouTubeApiLoader();
  const { win, doc, scripts } = makeFakeDom();
  const pre = doc.createElement("script");
  pre.src = YT_API_SRC;
  scripts.push(pre);

  const p = loadYouTubeIframeApi({ win, doc });
  assert.equal(scripts.length, 1); // no duplicate tag

  win.YT = { Player: function Player() {} };
  win.onYouTubeIframeAPIReady();
  await p;
  resetYouTubeApiLoader();
});

test("loadYouTubeIframeApi clears its cache on failure so a retry starts clean", async () => {
  resetYouTubeApiLoader();
  const { win, doc, scripts } = makeFakeDom();

  const p1 = loadYouTubeIframeApi({ win, doc });
  assert.equal(scripts.length, 1);
  scripts[0].dispatchEvent("error");
  await assert.rejects(p1, /failed to load/i);
  assert.equal(scripts.length, 0); // dead tag removed

  const p2 = loadYouTubeIframeApi({ win, doc });
  assert.equal(scripts.length, 1); // fresh attempt
  win.YT = { Player: function Player() {} };
  win.onYouTubeIframeAPIReady();
  await p2;
  resetYouTubeApiLoader();
});

test("loadYouTubeIframeApi rejects after a finite timeout instead of hanging forever", async (t) => {
  resetYouTubeApiLoader();
  t.mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
  const { win, doc } = makeFakeDom();

  const p = loadYouTubeIframeApi({ win, doc, timeoutMs: 5000 });
  const assertion = assert.rejects(p, /did not become ready/i);
  t.mock.timers.tick(5000);
  await assertion;

  t.mock.timers.reset();
  resetYouTubeApiLoader();
});
