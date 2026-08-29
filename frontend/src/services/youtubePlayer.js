/**
 * Framework-free lifecycle for the official YouTube IFrame Player API.
 *
 * The React hook (`useVideoPlayer`) is a thin adapter over this module; keeping
 * the imperative logic here means the lifecycle races that only show up on a
 * real desktop browser (delayed API load, StrictMode remount, hide/show, a
 * dependency change mid-load) can be unit-tested with a mocked `YT.Player`
 * instead of a headless browser.
 *
 * https://developers.google.com/youtube/iframe_api_reference
 */

export const YT_API_SRC = "https://www.youtube.com/iframe_api";

/** `YT.PlayerState` values, inlined so callers do not need `window.YT`. */
export const YT_PLAYER_STATE = {
  UNSTARTED: -1,
  ENDED: 0,
  PLAYING: 1,
  PAUSED: 2,
  BUFFERING: 3,
  CUED: 5,
};

/** Human-readable meaning for the numeric codes YouTube passes to `onError`. */
export const YT_ERROR_MESSAGES = {
  2: "YouTube rejected a player parameter (invalid video id or start time).",
  5: "The YouTube HTML5 player failed while trying to play this video.",
  100: "This video has been removed or marked private.",
  101: "The video owner does not allow it to be played in embedded players.",
  150: "The video owner does not allow it to be played in embedded players.",
  153: "YouTube blocked the embed because the request had no valid HTTP Referer / origin.",
};

/**
 * Map a raw `onError` payload (`event.data`) to `{ code, message }`.
 *
 * @param {number|string} code
 * @returns {{ code: number, message: string }}
 */
export function describeYouTubeError(code) {
  const n = Number(code);
  const known = Number.isFinite(n) ? YT_ERROR_MESSAGES[n] : undefined;
  return {
    code: Number.isFinite(n) ? n : 0,
    message: known || `YouTube reported player error code ${code}.`,
  };
}

function noop() {}

function safeCall(target, method) {
  try {
    return typeof target?.[method] === "function" ? target[method]() : null;
  } catch {
    return null;
  }
}

function iframeInfo(container) {
  const frame = container?.querySelector?.("iframe") || null;
  return { iframeUrl: frame?.src || null, iframeConnected: Boolean(frame?.isConnected) };
}

/* ------------------------------------------------------------------ *
 * API loader
 * ------------------------------------------------------------------ */

let apiLoad = null;

/** Drop the cached load promise (used after a failure and by tests). */
export function resetYouTubeApiLoader() {
  apiLoad = null;
}

/**
 * Load the IFrame Player API exactly once for all concurrent consumers.
 *
 * - keeps a singleton promise while the script is in flight;
 * - chains, never clobbers, an existing `window.onYouTubeIframeAPIReady`;
 * - reuses a `<script>` another consumer already added;
 * - polls as a safety net in case that callback is overwritten elsewhere;
 * - rejects (never hangs) after a finite timeout;
 * - clears the cache on rejection so a later call can retry from scratch.
 *
 * @param {{ timeoutMs?: number, win?: Window, doc?: Document }} [opts]
 * @returns {Promise<typeof window.YT>}
 */
export function loadYouTubeIframeApi({ timeoutMs = 10000, win, doc } = {}) {
  const w = win || (typeof window !== "undefined" ? window : null);
  const d = doc || (typeof document !== "undefined" ? document : null);
  if (!w || !d) {
    return Promise.reject(new Error("The YouTube IFrame API can only load in a browser environment."));
  }
  if (w.YT && typeof w.YT.Player === "function") return Promise.resolve(w.YT);
  if (apiLoad) return apiLoad.promise;

  let entry;
  const base = new Promise((resolve, reject) => {
    let settled = false;
    let pollTimer = 0;
    let failTimer = 0;
    const isReady = () => Boolean(w.YT && typeof w.YT.Player === "function");
    const clear = () => {
      clearInterval(pollTimer);
      clearTimeout(failTimer);
    };
    const finish = (fn, arg) => {
      if (settled) return;
      settled = true;
      clear();
      fn(arg);
    };
    const succeed = () => finish(resolve, w.YT);
    const fail = (message) => finish(reject, new Error(message));

    const prev = typeof w.onYouTubeIframeAPIReady === "function" ? w.onYouTubeIframeAPIReady : null;
    w.onYouTubeIframeAPIReady = () => {
      if (prev) {
        try {
          prev();
        } catch {
          /* another consumer's handler threw - not our problem */
        }
      }
      if (isReady()) succeed();
    };

    const existing = d.querySelector(`script[src="${YT_API_SRC}"]`);
    if (existing) {
      // Already requested (by us on an earlier mount, or by another consumer).
      // It may still be loading; the ready callback or the poll picks it up.
      existing.addEventListener(
        "error",
        () => {
          try {
            existing.remove();
          } catch {
            /* ignore */
          }
          fail("The YouTube IFrame API script failed to load.");
        },
        { once: true },
      );
    } else {
      const tag = d.createElement("script");
      tag.src = YT_API_SRC;
      tag.async = true;
      tag.addEventListener(
        "error",
        () => {
          try {
            tag.remove();
          } catch {
            /* ignore */
          }
          fail("The YouTube IFrame API script failed to load (offline, or blocked by an extension).");
        },
        { once: true },
      );
      (d.head || d.documentElement || d.body).appendChild(tag);
    }

    pollTimer = setInterval(() => {
      if (isReady()) succeed();
    }, 100);
    failTimer = setTimeout(() => {
      fail(
        `The YouTube IFrame API did not become ready within ${Math.round(
          Math.max(1000, timeoutMs) / 1000,
        )}s. Check the network tab for a blocked request to www.youtube.com.`,
      );
    }, Math.max(1000, timeoutMs));
  });

  const promise = base.catch((err) => {
    if (apiLoad === entry) apiLoad = null;
    throw err;
  });
  entry = { promise };
  apiLoad = entry;
  return promise;
}

/* ------------------------------------------------------------------ *
 * Player lifecycle
 * ------------------------------------------------------------------ */

function defaultCreatePlayer(YT, host, config) {
  return new YT.Player(host, config);
}

/**
 * Construct a `YT.Player` on `host` once the API is available and return a
 * `stop()` that tears exactly this instance down. Every async callback is
 * guarded so a late event from a disposed run cannot touch live state.
 *
 * `host` must already be attached to the document - `YT.Player` replaces it
 * with an `<iframe>`, so it must be a throwaway node, never a React-owned one.
 *
 * @returns {() => void} stop
 */
export function startYouTubePlayback({
  host,
  videoId,
  startSeconds = 0,
  autoplay = true,
  origin,
  timeoutMs,
  loadApi = loadYouTubeIframeApi,
  createPlayer = defaultCreatePlayer,
  onReady,
  onStateChange,
  onError,
  onAutoplayBlocked,
} = {}) {
  let disposed = false;
  let player = null;

  const guard = (fn) => (...args) => {
    if (!disposed) fn?.(...args);
  };

  Promise.resolve()
    .then(() => loadApi({ timeoutMs }))
    .then((YT) => {
      if (disposed) return;
      if (!host || host.isConnected === false) {
        onError?.({
          code: 0,
          message: "The video container was detached before the player finished loading.",
        });
        return;
      }
      player = createPlayer(YT, host, {
        videoId,
        playerVars: {
          start: Math.max(0, Math.floor(Number(startSeconds) || 0)),
          autoplay: autoplay ? 1 : 0,
          rel: 0,
          playsinline: 1,
          enablejsapi: 1,
          origin:
            origin || (typeof window !== "undefined" ? window.location.origin : undefined),
        },
        events: {
          onReady: guard((event) => onReady?.(event.target)),
          onStateChange: guard((event) => onStateChange?.(event.data, event.target)),
          onError: guard((event) => onError?.(describeYouTubeError(event.data))),
          onAutoplayBlocked: guard((event) => onAutoplayBlocked?.(event.target)),
        },
      });
    })
    .catch(
      guard((err) => {
        onError?.({
          code: 0,
          message: err instanceof Error ? err.message : "The YouTube player could not be created.",
        });
      }),
    );

  return function stop() {
    disposed = true;
    const doomed = player;
    player = null;
    try {
      doomed?.destroy?.();
    } catch {
      /* the iframe may already be gone */
    }
  };
}

/**
 * Imperative core of the React effect, extracted so its lifecycle contract is
 * directly testable:
 *
 *  - React owns the stable, always-empty `container`;
 *  - every run makes a fresh inner host node inside it (never reuse one
 *    `YT.Player` has already swapped for an `<iframe>`);
 *  - `playerRef.current` is set from `event.target` on ready and cleared on
 *    teardown **only if it still points at this run's player**, so a stale
 *    cleanup cannot erase a newer player;
 *  - cleanup destroys only this run's player and empties the container.
 *
 * @returns {() => void} cleanup
 */
export function runYouTubePlayerEffect({
  container,
  videoId,
  startSeconds = 0,
  autoplay = true,
  origin,
  timeoutMs,
  playerRef,
  loadApi,
  createPlayer,
  onReady,
  onStateChange,
  onAutoplayBlocked,
  onError,
  debug = noop,
}) {
  if (!container) return noop;

  // Hand YouTube a brand-new node every run. `container` stays React's.
  container.replaceChildren();
  const host = (container.ownerDocument || globalThis.document).createElement("div");
  host.setAttribute?.("data-yt-host", String(videoId || ""));
  container.appendChild(host);

  let disposed = false;
  let localPlayer = null;

  const stop = startYouTubePlayback({
    host,
    videoId,
    startSeconds,
    autoplay,
    origin,
    timeoutMs,
    ...(loadApi ? { loadApi } : {}),
    ...(createPlayer ? { createPlayer } : {}),
    onReady: (ytPlayer) => {
      if (disposed) return;
      localPlayer = ytPlayer;
      if (playerRef) playerRef.current = ytPlayer;
      const state = safeCall(ytPlayer, "getPlayerState");
      onReady?.({ player: ytPlayer, state });
      debug("onReady", {
        state,
        currentTime: safeCall(ytPlayer, "getCurrentTime"),
        loadedFraction: safeCall(ytPlayer, "getVideoLoadedFraction"),
        ...iframeInfo(container),
      });
    },
    onStateChange: (state, ytPlayer) => {
      if (disposed) return;
      onStateChange?.({ state, player: ytPlayer });
      debug("onStateChange", {
        state,
        currentTime: safeCall(ytPlayer, "getCurrentTime"),
        loadedFraction: safeCall(ytPlayer, "getVideoLoadedFraction"),
      });
    },
    onAutoplayBlocked: () => {
      if (disposed) return;
      onAutoplayBlocked?.();
      debug("onAutoplayBlocked", { note: "browser requires a user gesture before playback" });
    },
    onError: (info) => {
      if (disposed) return;
      onError?.(info);
      debug("onError", { ...info, ...iframeInfo(container) });
    },
  });

  return function cleanup() {
    disposed = true;
    stop();
    if (playerRef && playerRef.current === localPlayer) playerRef.current = null;
    try {
      container.replaceChildren();
    } catch {
      /* ignore */
    }
    debug("cleanup", {});
  };
}
