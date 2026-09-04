import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  CloseOutlined,
  DeleteOutlined,
  ExportOutlined,
  MessageOutlined,
  PlusSquareOutlined,
  VideoCameraOutlined,
  OrderedListOutlined,
} from "@ant-design/icons";
import { fmtDur, toTimecode } from "../../shared/format";
import { keyframeImgProps } from "../../shared/imageFallback";
import { getFramePool } from "../../mocks/searchEngine";
import { fetchVideoTimeline } from "../../shared/adapters";
import { buildVideoPlayback, youtubeVideoId } from "../../services/videoPlayback";
import { buildCaptureCandidate, captureFrame, fetchPlayback } from "../../services/videoCapture";
import useDialogFocus from "../../hooks/useDialogFocus";
import useVideoPlayer from "../../hooks/useVideoPlayer";

export default function ReviewOverlay({ item, results, isKept, onClose, onNavigate, onSelect, onToggleKeep, onRemove, onPivot, onAsk, onExportThis, onCapture, replaceCtx, onReplaceEventFrame }) {
  const [compareId, setCompareId] = useState(null);
  const [stripMode, setStripMode] = useState("timeline"); // "timeline" | "results"
  const [timeline, setTimeline] = useState([]);
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  const [showVideo, setShowVideo] = useState(false);
  const [playback, setPlayback] = useState(null);
  const [playbackError, setPlaybackError] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const [captureNote, setCaptureNote] = useState(null);
  const backRef = useRef(null);
  const dialogRef = useDialogFocus(backRef);
  const hydratedItem = useMemo(() => {
    if (!item || !timeline.length) return item;
    const match = timeline.find((candidate) =>
      candidate.id === item.id
      || (candidate.frameKey && item.frameKey && candidate.frameKey === item.frameKey)
      || (candidate.frameName && item.frameName && candidate.frameName === item.frameName)
    );
    return match ? { ...item, ...match, rank: item.rank, score: item.score } : item;
  }, [item, timeline]);
  // The backend playback endpoint is the source of truth for the watch URL.
  // Search enrichment may not carry it (e.g. media-info served from a zip the
  // retriever doesn't read), so fall back to the metadata we just fetched.
  const videoItem = useMemo(() => {
    if (!hydratedItem) return hydratedItem;
    if (!playback?.watchUrl) return hydratedItem;
    // Backend watch_url is authoritative; item.link is often the string
    // "undefined" from upstream normalization, so don't prefer it.
    return { ...hydratedItem, link: playback.watchUrl };
  }, [hydratedItem, playback?.watchUrl]);
  const videoPlayback = useMemo(
    () => buildVideoPlayback(videoItem, playback?.playbackOffsetSeconds || 0),
    [videoItem, playback?.playbackOffsetSeconds],
  );
  const playerStart = playback?.startSeconds != null ? playback.startSeconds : videoPlayback?.start || 0;
  const player = useVideoPlayer({
    type: videoPlayback?.type === "youtube" ? "youtube" : "video",
    youtubeId: videoPlayback ? youtubeVideoId(videoPlayback.url) : "",
    url: videoPlayback?.url || "",
    start: playerStart,
    active: Boolean(showVideo && videoPlayback),
  });
  const captureFps = playback?.fps || hydratedItem?.fps || 25;

  useEffect(() => {
    let active = true;
    if (!item?.videoKey || item.videoKey === "unknown-video") {
      setTimeline([]);
      return;
    }

    if (!item.real) {
      const pool = getFramePool();
      const mockNeighbors = pool.filter((f) => f.videoKey === item.videoKey).sort((a, b) => a.timestamp - b.timestamp);
      setTimeline(mockNeighbors);
      return;
    }

    setLoadingTimeline(true);
    fetchVideoTimeline(item.videoKey, item.frameKey || item.id, 60)
      .then((frames) => {
        if (active) {
          setTimeline(frames || []);
          setLoadingTimeline(false);
        }
      })
      .catch(() => {
        if (active) setLoadingTimeline(false);
      });

    return () => {
      active = false;
    };
  }, [item?.videoKey, item?.frameKey, item?.id, item?.real]);

  useEffect(() => {
    setCompareId(null);
    setShowVideo(false);
    setCaptureNote(null);
  }, [item?.id]);

  // Load backend playback metadata (watch_url, fps, offset, start time for the
  // frame under review). Capture stays disabled until this resolves.
  useEffect(() => {
    let active = true;
    setPlayback(null);
    setPlaybackError(null);

    const videoId = item?.videoKey;
    if (!videoId || videoId === "unknown-video" || !item?.real) return undefined;

    const frameIdx = item?.submissionFrameId ?? item?.frameKey ?? null;
    fetchPlayback(videoId, frameIdx)
      .then((meta) => {
        if (active) setPlayback(meta);
      })
      .catch((err) => {
        if (active) setPlaybackError(err instanceof Error ? err.message : "Playback metadata unavailable.");
      });

    return () => {
      active = false;
    };
  }, [item?.videoKey, item?.submissionFrameId, item?.frameKey, item?.real]);

  const canCapture = Boolean(
    (onCapture || onReplaceEventFrame) && videoPlayback && showVideo && player.ready && playback && !playbackError && !capturing,
  );

  const handleCapture = async () => {
    if (!canCapture) return;
    const currentTime = player.getCurrentTime();
    if (currentTime == null) {
      setCaptureNote({ type: "error", text: "Could not read the player time yet - try again." });
      return;
    }
    setCapturing(true);
    setCaptureNote(null);
    player.pause();
    try {
      const result = await captureFrame(videoItem.videoKey, currentTime);
      const candidate = buildCaptureCandidate(videoItem, result);
      const previewSuffix = candidate.hasPreview ? "" : " - preview unavailable";

      // Storyboard "replace event" flow: the captured frame becomes event N,
      // subject to the same window as a timeline pick (same video is implied,
      // timestamp must sit between the neighbouring events).
      if (replaceCtx) {
        const ts = Number(result.sourceTimeSeconds);
        if (!(ts > replaceCtx.minTs && ts < replaceCtx.maxTs) || !Number.isFinite(Number(result.frameIdx))) {
          setCaptureNote({
            type: "error",
            text: `Captured ${toTimecode(ts, result.fps || captureFps)} is outside event ${replaceCtx.eventIndex}'s window - scrub between the neighbouring events, then capture.`,
          });
          return;
        }
        onReplaceEventFrame?.(candidate);
        setCaptureNote({
          type: "ok",
          text: `Captured frame ${result.frameIdx} -> event ${replaceCtx.eventIndex}${previewSuffix}`,
        });
        onClose();
        return;
      }

      const outcome = onCapture?.(candidate, result);
      const added = outcome === undefined ? true : Boolean(outcome);
      setCaptureNote({
        type: added ? "ok" : "dupe",
        text: added
          ? `Captured frame ${result.frameIdx} (${toTimecode(result.sourceTimeSeconds, result.fps || captureFps)}) - added to tray${previewSuffix}`
          : `Frame ${result.frameIdx} is already in the tray`,
      });
    } catch (err) {
      setCaptureNote({
        type: "error",
        text: err instanceof Error ? err.message : "Capture failed.",
      });
    } finally {
      setCapturing(false);
    }
  };

  if (!item) return null;

  const strip = stripMode === "timeline" && timeline.length > 0 ? timeline : (results || []);
  const seq = strip;
  const curIdx = seq.findIndex((x) => x.id === item.id || (x.frameKey && x.frameKey === item.frameKey));
  const compareItem = compareId ? (results || []).find((r) => r.id === compareId) : null;
  const atStart = curIdx <= 0;
  const atEnd = curIdx === -1 || curIdx >= seq.length - 1;

  // "Replace event frame": the frame currently under review becomes event N of
  // an edited temporal sequence, but only if it stays in the same video and
  // between the neighbouring events' timestamps.
  const replaceCandidate = hydratedItem || item;
  const replaceFrameId = Number(replaceCandidate?.submissionFrameId ?? replaceCandidate?.backend?.frame_idx);
  const replaceSameVideo = Boolean(replaceCtx) && replaceCandidate?.videoKey === replaceCtx?.videoKey;
  const replaceInWindow =
    Boolean(replaceCtx) &&
    Number(replaceCandidate?.timestamp) > replaceCtx.minTs &&
    Number(replaceCandidate?.timestamp) < replaceCtx.maxTs;
  const canReplace = Boolean(replaceCtx) && replaceSameVideo && replaceInWindow && Number.isFinite(replaceFrameId);
  const replaceHint = !replaceCtx
    ? ""
    : !replaceSameVideo
      ? `Replacement must stay in ${replaceCtx.videoKey}`
      : !Number.isFinite(replaceFrameId)
        ? "This frame has no resolvable frame index"
        : !replaceInWindow
          ? "Replacement must sit between the neighbouring events' timestamps"
          : `Use this frame for event ${replaceCtx.eventIndex}`;
  const filmstripLabel = loadingTimeline
    ? `Loading timeline - ${item.videoKey}`
    : stripMode === "timeline"
      ? `Video timeline - ${item.videoKey} (${timeline.length} keyframes)`
      : `Search query results (${results?.length || 0} frames)`;
  const handleStep = (dir) => {
    if (curIdx === -1) {
      if (dir > 0 && seq.length > 0) onSelect(seq[0]);
      return;
    }
    const nextIdx = Math.max(0, Math.min(seq.length - 1, curIdx + dir));
    if (seq[nextIdx] && seq[nextIdx].id !== item.id) {
      onSelect(seq[nextIdx]);
    } else {
      onNavigate(dir);
    }
  };


  return (
    <div ref={dialogRef} className="ws-review" role="dialog" aria-modal="true" aria-label={`Frame review: ${item.frameName}`} tabIndex={-1}>
      <header className="ws-review-head">
        <button ref={backRef} className="ws-btn small" onClick={onClose} title="Back to grid (Esc)">
          <ArrowLeftOutlined /> Back
        </button>
        <div className="ws-review-title">
          Frame Review
          <span className="ws-review-ctx">
            {item.videoKey} - {item.camera}
          </span>
        </div>
        <span className="ws-review-pos">{curIdx >= 0 ? `${curIdx + 1} / ${seq.length}` : "-"}</span>
        <div className="ws-review-head-actions">
          <button className="ws-btn small" onClick={() => handleStep(-1)} title="Previous (Left)" disabled={atStart}>
            <ArrowLeftOutlined /> Prev
          </button>
          <button className="ws-btn small" onClick={() => handleStep(1)} title="Next (Right)" disabled={atEnd}>
            <ArrowRightOutlined /> Next
          </button>
          <button className={`ws-btn small ${isKept ? "" : "primary"}`} onClick={() => onToggleKeep(item)} title="Keep / unkeep (Space)">
            {isKept ? "Kept" : "Keep"}
          </button>
          {replaceCtx ? null : (
            <button className="ws-btn small" onClick={() => onPivot(item)} title="Similar-image pivot">
              <PlusSquareOutlined /> Similar
            </button>
          )}
          <button className="ws-btn small" onClick={() => onAsk(item)} title="Ask the copilot about this frame">
            <MessageOutlined /> Ask
          </button>
          {videoPlayback ? (
            <button className="ws-btn small" onClick={() => setShowVideo((value) => !value)} title="Toggle YouTube/video stream">
              <VideoCameraOutlined /> {showVideo ? "Show frame" : "Play video"}
            </button>
          ) : null}
          {replaceCtx ? (
            <button
              className="ws-btn small primary"
              onClick={() => {
                onReplaceEventFrame?.(replaceCandidate);
                onClose();
              }}
              disabled={!canReplace}
              title={replaceHint}
            >
              <OrderedListOutlined /> Replace event {replaceCtx.eventIndex}
            </button>
          ) : (
            <button className="ws-btn small" onClick={() => onRemove(item)} title="Remove from results (Delete)">
              <DeleteOutlined /> Remove
            </button>
          )}
        </div>
      </header>

      <div className="ws-review-body">
        <div className="ws-review-stage">
          <div className="ws-review-viewport">
            {compareItem ? (
              <div className="ws-review-cmp">
                <div className="ws-review-cmp-item">
                  <span className="ws-review-cmp-label">CURRENT - {item.frameName}</span>
                  <img src={item.image} alt={item.frameName} {...keyframeImgProps} />
                </div>
                <div className="ws-review-cmp-item">
                  <span className="ws-review-cmp-label">COMPARE - {compareItem.frameName}</span>
                  <img src={compareItem.image} alt={compareItem.frameName} {...keyframeImgProps} />
                </div>
              </div>
            ) : showVideo && videoPlayback ? (
              <div className="ws-review-player">
                <div className="ws-review-player-frame">
                  {videoPlayback.type === "youtube" ? (
                    <div className="ws-review-yt" ref={player.hostRef} />
                  ) : (
                    <video ref={player.hostRef} src={videoPlayback.url} controls autoPlay />
                  )}
                  {videoPlayback.type === "youtube" && player.ready && player.autoplayBlocked && !player.playing ? (
                    <button
                      type="button"
                      className="ws-review-play-shade"
                      onClick={() => player.play()}
                      title="Your browser blocked autoplay - click to start playback"
                    >
                      <VideoCameraOutlined />
                      <span>Click to play</span>
                      <small>Autoplay was blocked by the browser</small>
                    </button>
                  ) : null}
                </div>
                <div className="ws-review-capture-bar">
                  <button
                    className="ws-btn small primary"
                    type="button"
                    onClick={handleCapture}
                    disabled={!canCapture}
                    title={
                      playbackError
                        ? playbackError
                        : !player.ready
                          ? "Waiting for the player to be ready"
                          : "Capture the frame at the current playback position"
                    }
                  >
                    <VideoCameraOutlined />{" "}
                    {capturing
                      ? "Capturing..."
                      : replaceCtx
                        ? `Capture -> event ${replaceCtx.eventIndex}`
                        : "Capture frame"}
                  </button>
                  <span className="ws-review-capture-hint">
                    {playbackError
                      ? `Capture unavailable: ${playbackError}`
                      : player.error
                        ? player.error
                        : !player.ready
                          ? "Loading player..."
                          : player.autoplayBlocked && !player.playing
                            ? "Autoplay blocked - click the video to play, then scrub to the exact moment"
                            : playback
                              ? `${playback.fps} fps${playback.playbackOffsetSeconds ? ` - offset ${playback.playbackOffsetSeconds}s` : ""}${player.playing ? "" : " - paused"}`
                              : "Loading metadata..."}
                  </span>
                  {captureNote ? (
                    <span className={`ws-review-capture-note ${captureNote.type}`} role="status">
                      {captureNote.text}
                    </span>
                  ) : null}
                </div>
              </div>
            ) : (
              <button
                className={`ws-review-media-button ${videoPlayback ? "playable" : ""}`}
                type="button"
                onClick={() => videoPlayback && setShowVideo(true)}
                title={videoPlayback ? "Play video from this timestamp" : hydratedItem.frameName}
              >
                {hydratedItem.image ? (
                  <img className="ws-review-img" src={hydratedItem.image} alt={hydratedItem.frameName} {...keyframeImgProps} />
                ) : (
                  <div className="ws-review-img ws-review-img-missing" role="img" aria-label="Exact preview unavailable">
                    <VideoCameraOutlined />
                    <span>Preview unavailable</span>
                    <small>{hydratedItem.previewError || "The captured frame is still export-ready."}</small>
                  </div>
                )}
                {videoPlayback ? (
                  <span className="ws-review-media-play">
                    <VideoCameraOutlined />
                    Play from {hydratedItem.timecode}
                  </span>
                ) : null}
              </button>
            )}
            <button className="ws-review-nav prev" onClick={() => handleStep(-1)} title="Previous frame (Left)" disabled={atStart}>
              <ArrowLeftOutlined />
            </button>
            <button className="ws-review-nav next" onClick={() => handleStep(1)} title="Next frame (Right)" disabled={atEnd}>
              <ArrowRightOutlined />
            </button>
          </div>

          <div className="ws-filmstrip">
            <div className="ws-filmstrip-head">
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <span className="ws-filmstrip-tag">
                  {filmstripLabel}
                </span>
                {timeline.length > 0 ? (
                  <button
                    className="ws-btn small"
                    style={{ fontSize: "11px", padding: "1px 8px", background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: "4px" }}
                    onClick={() => setStripMode((m) => (m === "timeline" ? "results" : "timeline"))}
                    title="Toggle strip mode between video sequence and search results"
                  >
                    {stripMode === "timeline" ? (
                      <><OrderedListOutlined /> Show Search Results</>
                    ) : (
                      <><VideoCameraOutlined /> Show {item.videoKey} Timeline</>
                    )}
                  </button>
                ) : null}
              </div>
              {compareItem ? (
                <button className="ws-filmstrip-clear" onClick={() => setCompareId(null)}>
                  <CloseOutlined /> End compare
                </button>
              ) : null}
            </div>
            <div className="ws-filmstrip-track">
              {strip.map((f) => (
                <button
                  key={f.id}
                  className={`ws-film ${f.id === item.id || (f.frameKey && f.frameKey === item.frameKey) ? "current" : ""}`}
                  onClick={() => onSelect(f)}
                  title={f.frameName}
                >
                  <img src={f.image} alt={f.frameName} {...keyframeImgProps} />
                  <span className="ws-film-tc">{f.timecode}</span>
                  {f.id === item.id || (f.frameKey && f.frameKey === item.frameKey) ? <span className="ws-film-cur">CURRENT</span> : null}
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="ws-review-side">
          <div className="ws-review-side-title">Frame details</div>
          <dl className="ws-review-meta">
            <div className="ws-review-row"><dt className="k">Video</dt><dd className="v cyan">{hydratedItem.videoKey}</dd></div>
            <div className="ws-review-row"><dt className="k">Frame ID</dt><dd className="v">{hydratedItem.frameKey || item.id}</dd></div>
            <div className="ws-review-row"><dt className="k">Global ID</dt><dd className="v">#{hydratedItem.globalFrameId}</dd></div>
            <div className="ws-review-row"><dt className="k">Timestamp</dt><dd className="v amber">{hydratedItem.timecode} - {fmtDur(hydratedItem.timestamp)}</dd></div>
            {videoPlayback ? (
              <div className="ws-review-row">
                <dt className="k">Video stream</dt>
                <dd className="v cyan">
                  <a href={videoPlayback.url} target="_blank" rel="noreferrer">
                    open source
                  </a>
                </dd>
              </div>
            ) : null}
            <div className="ws-review-row"><dt className="k">Rank</dt><dd className="v">{item.rank != null ? item.rank : "-"}</dd></div>
            <div className="ws-review-row"><dt className="k">Score</dt><dd className="v">{item.score != null ? `${Math.round(item.score * 100)}%` : "-"}</dd></div>
            <div className="ws-review-row"><dt className="k">Camera</dt><dd className="v">{hydratedItem.camera}</dd></div>
            <div className="ws-review-row"><dt className="k">Source</dt><dd className="v">{hydratedItem.real ? "broadcast feed" : `${hydratedItem.folderKey} / ${hydratedItem.videoKey}`}</dd></div>
          </dl>
          <p className="ws-review-note">Score &amp; provenance are placeholders until the backend stabilizes.</p>

          <div className="ws-review-cmp-field">
            <label className="ws-param-label">Compare with another result</label>
            <select value={compareId || ""} onChange={(e) => setCompareId(e.target.value || null)}>
              <option value="">- select a frame -</option>
              {(results || []).filter((r) => r.id !== item.id).map((r) => (
                <option key={r.id} value={r.id}>
                  #{String(r.rank ?? "-")} - {r.frameName} - {r.timecode}
                </option>
              ))}
            </select>
          </div>

          <div className="ws-review-actions">
            {item.answer ? <div className="ws-answer">QA: {item.answer}</div> : null}
            <button className="ws-btn small" onClick={() => onExportThis(item)}>
              <ExportOutlined /> Export this frame
            </button>
            <button className="ws-btn small" onClick={onClose}>
              Close review
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
