import { useCallback, useEffect, useRef, useState } from "react";
import {
  YT_PLAYER_STATE,
  loadYouTubeIframeApi,
  runYouTubePlayerEffect,
} from "../services/youtubePlayer.js";

/**
 * Drive a YouTube IFrame player (or a plain HTML5 <video>) for the Review
 * overlay so the Capture button can read an accurate current time.
 *
 * YouTube path: the imperative lifecycle lives in `services/youtubePlayer.js`.
 * This hook only wires it to React state/refs. React owns a stable, always
 * empty outer container (`hostRef`); each effect run creates a fresh inner
 * node inside it and hands *that* to `YT.Player`, which replaces it with an
 * <iframe>. We never reuse a node YouTube has already swallowed.
 * `enablejsapi=1` + an explicit `origin` are required for `getCurrentTime()`
 * / `pauseVideo()` to work across origins.
 * https://developers.google.com/youtube/iframe_api_reference
 *
 * HTML5 path: `hostRef` is the <video> element itself and we read
 * `currentTime` / call `pause()` directly.
 *
 * @param {object}  opts
 * @param {"youtube"|"video"} opts.type
 * @param {string}  opts.youtubeId  - required when type === "youtube"
 * @param {string}  opts.url        - required when type === "video"
 * @param {number}  opts.start      - start offset in seconds
 * @param {boolean} opts.active     - mount/enable the player
 * @returns {{
 *   hostRef: React.RefObject<HTMLElement>,
 *   ready: boolean,
 *   error: string|null,
 *   playerState: number,
 *   playing: boolean,
 *   autoplayBlocked: boolean,
 *   getCurrentTime: () => number|null,
 *   pause: () => void,
 *   play: () => void,
 *   seekTo: (seconds: number) => void,
 * }}
 */

const DEV = Boolean(import.meta.env?.DEV);

// Dev-only instrumentation: logs onReady / onStateChange / onAutoplayBlocked /
// onError plus getPlayerState / getCurrentTime / getVideoLoadedFraction and the
// iframe URL + connection status. Left `undefined` in production so the effect
// skips building the detail payload entirely.
const debugLog = DEV
  ? (event, detail) => console.info(`[useVideoPlayer] ${event}`, detail)
  : undefined;

export default function useVideoPlayer({ type, youtubeId, url, start = 0, active = true } = {}) {
  // YouTube: React-owned stable outer box. HTML5: the <video> element.
  const hostRef = useRef(null);
  // The current, ready YT.Player - or null when nothing is live.
  const playerRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [playerState, setPlayerState] = useState(YT_PLAYER_STATE.UNSTARTED);
  const [autoplayBlocked, setAutoplayBlocked] = useState(false);

  const startSeconds = Math.max(0, Math.floor(Number(start) || 0));
  const isYouTube = type === "youtube";

  useEffect(() => {
    setReady(false);
    setError(null);
    setAutoplayBlocked(false);
    setPlayerState(YT_PLAYER_STATE.UNSTARTED);

    if (!active) return undefined;

    // ---- HTML5 <video> ------------------------------------------------
    if (!isYouTube) {
      const video = hostRef.current;
      if (!video) return undefined;
      const onLoaded = () => {
        try {
          if (startSeconds > 0) video.currentTime = startSeconds;
        } catch {
          /* seeking before metadata is fine to ignore */
        }
        setReady(true);
      };
      const onError = () => setError("The video failed to load.");
      video.addEventListener("loadedmetadata", onLoaded);
      video.addEventListener("error", onError);
      if (video.readyState >= 1) onLoaded();
      return () => {
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
      };
    }

    // ---- YouTube IFrame Player --------------------------------------
    if (!youtubeId) {
      setError("No YouTube video id was resolved for this result.");
      return undefined;
    }

    const container = hostRef.current;
    if (!container) return undefined;

    const cleanup = runYouTubePlayerEffect({
      container,
      videoId: youtubeId,
      startSeconds,
      autoplay: true,
      origin: window.location.origin,
      playerRef,
      loadApi: loadYouTubeIframeApi,
      onReady: ({ state }) => {
        setReady(true);
        if (state != null) setPlayerState(state);
      },
      onStateChange: ({ state }) => {
        setPlayerState(state);
        if (state === YT_PLAYER_STATE.PLAYING || state === YT_PLAYER_STATE.BUFFERING) {
          setAutoplayBlocked(false);
        }
      },
      onAutoplayBlocked: () => setAutoplayBlocked(true),
      onError: ({ code, message }) => {
        setError(code ? `YouTube error ${code}: ${message}` : message);
      },
      debug: debugLog,
    });

    return () => {
      cleanup();
      setReady(false);
      setAutoplayBlocked(false);
      setPlayerState(YT_PLAYER_STATE.UNSTARTED);
    };
  }, [isYouTube, youtubeId, url, startSeconds, active]);

  const getCurrentTime = useCallback(() => {
    if (isYouTube) {
      const p = playerRef.current;
      if (p && typeof p.getCurrentTime === "function") {
        try {
          const t = Number(p.getCurrentTime());
          return Number.isFinite(t) ? t : null;
        } catch {
          return null;
        }
      }
      return null;
    }
    const t = Number(hostRef.current?.currentTime);
    return Number.isFinite(t) ? t : null;
  }, [isYouTube]);

  const pause = useCallback(() => {
    if (isYouTube) {
      try {
        playerRef.current?.pauseVideo?.();
      } catch {
        /* ignore */
      }
      return;
    }
    try {
      hostRef.current?.pause?.();
    } catch {
      /* ignore */
    }
  }, [isYouTube]);

  const play = useCallback(() => {
    if (isYouTube) {
      try {
        playerRef.current?.playVideo?.();
      } catch {
        /* ignore */
      }
      // Optimistic: a real click satisfies the autoplay policy. onStateChange
      // will confirm, or re-raise the blocked flag if it somehow still fails.
      setAutoplayBlocked(false);
      return;
    }
    try {
      hostRef.current?.play?.();
    } catch {
      /* ignore */
    }
  }, [isYouTube]);

  const seekTo = useCallback(
    (seconds) => {
      const target = Math.max(0, Number(seconds) || 0);
      if (isYouTube) {
        try {
          playerRef.current?.seekTo?.(target, true);
        } catch {
          /* ignore */
        }
        return;
      }
      try {
        if (hostRef.current) hostRef.current.currentTime = target;
      } catch {
        /* ignore */
      }
    },
    [isYouTube],
  );

  return {
    hostRef,
    ready,
    error,
    playerState,
    playing: playerState === YT_PLAYER_STATE.PLAYING,
    autoplayBlocked,
    getCurrentTime,
    pause,
    play,
    seekTo,
  };
}
