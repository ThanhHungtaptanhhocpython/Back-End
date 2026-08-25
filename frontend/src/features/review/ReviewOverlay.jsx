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
import { fmtDur } from "../../shared/format";
import { getFramePool } from "../../mocks/searchEngine";
import { fetchVideoTimeline } from "../../shared/adapters";
import { buildVideoPlayback } from "../../services/videoPlayback";
import useDialogFocus from "../../hooks/useDialogFocus";

export default function ReviewOverlay({ item, results, isKept, onClose, onNavigate, onSelect, onToggleKeep, onRemove, onPivot, onAsk, onExportThis }) {
  const [compareId, setCompareId] = useState(null);
  const [stripMode, setStripMode] = useState("timeline"); // "timeline" | "results"
  const [timeline, setTimeline] = useState([]);
  const [loadingTimeline, setLoadingTimeline] = useState(false);
  const [showVideo, setShowVideo] = useState(false);
  const backRef = useRef(null);
  const dialogRef = useDialogFocus(backRef);
  const videoPlayback = useMemo(() => buildVideoPlayback(item), [item]);

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
  }, [item?.videoKey, item?.real]);

  useEffect(() => {
    setCompareId(null);
    setShowVideo(false);
  }, [item?.id]);

  if (!item) return null;

  const strip = stripMode === "timeline" && timeline.length > 0 ? timeline : (results || []);
  const seq = strip;
  const curIdx = seq.findIndex((x) => x.id === item.id || (x.frameKey && x.frameKey === item.frameKey));
  const compareItem = compareId ? (results || []).find((r) => r.id === compareId) : null;
  const atStart = curIdx <= 0;
  const atEnd = curIdx === -1 || curIdx >= seq.length - 1;
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
          <button className="ws-btn small" onClick={() => onPivot(item)} title="Similar-image pivot">
            <PlusSquareOutlined /> Similar
          </button>
          <button className="ws-btn small" onClick={() => onAsk(item)} title="Ask the copilot about this frame">
            <MessageOutlined /> Ask
          </button>
          {videoPlayback ? (
            <button className="ws-btn small" onClick={() => setShowVideo((value) => !value)} title="Toggle YouTube/video stream">
              <VideoCameraOutlined /> {showVideo ? "Show frame" : "Play video"}
            </button>
          ) : null}
          <button className="ws-btn small" onClick={() => onRemove(item)} title="Remove from results (Delete)">
            <DeleteOutlined /> Remove
          </button>
        </div>
      </header>

      <div className="ws-review-body">
        <div className="ws-review-stage">
          <div className="ws-review-viewport">
            {compareItem ? (
              <div className="ws-review-cmp">
                <div className="ws-review-cmp-item">
                  <span className="ws-review-cmp-label">CURRENT - {item.frameName}</span>
                  <img src={item.image} alt={item.frameName} />
                </div>
                <div className="ws-review-cmp-item">
                  <span className="ws-review-cmp-label">COMPARE - {compareItem.frameName}</span>
                  <img src={compareItem.image} alt={compareItem.frameName} />
                </div>
              </div>
            ) : showVideo && videoPlayback ? (
              <div className="ws-review-player">
                {videoPlayback.type === "youtube" ? (
                  <iframe
                    src={videoPlayback.embedUrl}
                    title={`${item.videoKey} YouTube stream`}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    allowFullScreen
                  />
                ) : (
                  <video src={videoPlayback.url} controls autoPlay />
                )}
              </div>
            ) : (
              <button
                className={`ws-review-media-button ${videoPlayback ? "playable" : ""}`}
                type="button"
                onClick={() => videoPlayback && setShowVideo(true)}
                title={videoPlayback ? "Play video from this timestamp" : item.frameName}
              >
                <img className="ws-review-img" src={item.image} alt={item.frameName} />
                {videoPlayback ? (
                  <span className="ws-review-media-play">
                    <VideoCameraOutlined />
                    Play from {item.timecode}
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
                  <img src={f.image} alt={f.frameName} loading="lazy" />
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
            <div className="ws-review-row"><dt className="k">Video</dt><dd className="v cyan">{item.videoKey}</dd></div>
            <div className="ws-review-row"><dt className="k">Frame ID</dt><dd className="v">{item.id}</dd></div>
            <div className="ws-review-row"><dt className="k">Global ID</dt><dd className="v">#{item.globalFrameId}</dd></div>
            <div className="ws-review-row"><dt className="k">Timestamp</dt><dd className="v amber">{item.timecode} - {fmtDur(item.timestamp)}</dd></div>
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
            <div className="ws-review-row"><dt className="k">Camera</dt><dd className="v">{item.camera}</dd></div>
            <div className="ws-review-row"><dt className="k">Source</dt><dd className="v">{item.real ? "broadcast feed" : `${item.folderKey} / ${item.videoKey}`}</dd></div>
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
