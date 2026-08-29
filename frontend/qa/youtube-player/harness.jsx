/* eslint-disable react-refresh/only-export-components */
/**
 * Manual + automated harness for `useVideoPlayer`.
 *
 * Served by the Vite dev server at /qa/youtube-player/harness.html. It drives
 * the real hook against the real YouTube IFrame API with NO backend, so the
 * lifecycle can be checked in a real desktop Chromium (see
 * `qa/youtube-player/verify-youtube.mjs`).
 *
 * Query params:
 *   ?v=<id>       initial video id           (default: M7lc1UVf-VE)
 *   ?start=<sec>  start offset               (default: 0)
 *   ?show=1       mount the player on load
 */
import React, { StrictMode, useRef, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import useVideoPlayer from "../../src/hooks/useVideoPlayer.js";

const params = new URLSearchParams(location.search);
const GOOD = "M7lc1UVf-VE"; // YouTube's own IFrame API sample video
const ALT = params.get("v") || "mjy8h-iT-ms";

const logLines = [];
function pushLog(kind, detail) {
  const line = `${new Date().toISOString().slice(11, 23)} ${kind} ${JSON.stringify(detail ?? {})}`;
  logLines.push(line);
  window.__harnessLog = logLines;
}
// Mirror the hook's dev console.info instrumentation into the on-page log.
const origInfo = console.info.bind(console);
console.info = (...args) => {
  if (typeof args[0] === "string" && args[0].startsWith("[useVideoPlayer]")) {
    pushLog(args[0], args[1]);
  }
  origInfo(...args);
};

function Panel() {
  const [videoId, setVideoId] = useState(params.get("v") || GOOD);
  const [start, setStart] = useState(Number(params.get("start") || 0));
  const [show, setShow] = useState(params.get("show") === "1");
  const [, force] = useState(0);

  const player = useVideoPlayer({
    type: "youtube",
    youtubeId: videoId,
    url: `https://www.youtube.com/watch?v=${videoId}`,
    start,
    active: show,
  });

  // Imperative hooks for the automated verifier (test/manual/verify-youtube.mjs).
  useEffect(() => {
    window.__play = () => player.play();
    window.__pause = () => player.pause();
    window.__setShow = (v) => setShow(Boolean(v));
    window.__setVideo = (id) => setVideoId(String(id));
    window.__setStart = (n) => setStart(Number(n) || 0);
  });

  const timesRef = useRef([]);
  useEffect(() => {
    const id = setInterval(() => {
      const t = player.getCurrentTime();
      timesRef.current.push(t);
      window.__harnessState = {
        ready: player.ready,
        error: player.error,
        playerState: player.playerState,
        playing: player.playing,
        autoplayBlocked: player.autoplayBlocked,
        currentTime: t,
        times: timesRef.current.slice(-10),
        iframeCount: document.querySelectorAll("#frame iframe").length,
      };
      force((n) => n + 1);
    }, 500);
    return () => clearInterval(id);
  }, [player]);

  const s = window.__harnessState || {};
  return (
    <div>
      <div className="row">
        <div>
          <button id="btn-show" onClick={() => setShow((v) => !v)}>{show ? "Hide" : "Show"} video</button>
          <button id="btn-good" onClick={() => setVideoId(GOOD)}>id: known-good</button>
          <button id="btn-alt" onClick={() => setVideoId(ALT)}>id: {ALT}</button>
          <button id="btn-bad" onClick={() => setVideoId("00000000000")}>id: invalid</button>
          <button id="btn-start" onClick={() => setStart((x) => (x ? 0 : 90))}>toggle start (now {start})</button>
        </div>
      </div>
      <p>
        <span className="k">ready</span>={String(player.ready)}{"  "}
        <span className="k">state</span>={player.playerState}{"  "}
        <span className="k">playing</span>={String(player.playing)}{"  "}
        <span className="k">autoplayBlocked</span>={String(player.autoplayBlocked)}{"  "}
        <span className="k">error</span>={String(player.error)}{"  "}
        <span className="k">t</span>={String(player.getCurrentTime())}{"  "}
        <span className="k">iframes</span>={s.iframeCount ?? 0}
      </p>
      <div id="frame" style={{ position: "relative", aspectRatio: "16 / 9", background: "#000" }}>
        {show ? <div className="yt-outer" ref={player.hostRef} style={{ width: "100%", height: "100%" }} /> : null}
        {show && player.ready && player.autoplayBlocked && !player.playing ? (
          <button
            id="btn-clicktoplay"
            onClick={() => player.play()}
            style={{ position: "absolute", inset: 0, background: "rgba(2,6,23,.6)", color: "#fff", border: 0 }}
          >
            Click to play (autoplay blocked)
          </button>
        ) : null}
      </div>
      <pre id="log">{(window.__harnessLog || []).join("\n")}</pre>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Panel />
  </StrictMode>,
);
