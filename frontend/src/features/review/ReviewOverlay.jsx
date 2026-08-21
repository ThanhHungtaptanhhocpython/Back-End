import { useEffect, useRef, useState } from "react";
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
import useDialogFocus from "../../hooks/useDialogFocus";

export default function ReviewOverlay({ item, results, isKept, onClose, onNavigate, onSelect, onToggleKeep, onRemove, onPivot, onAsk, onExportThis }) {
  const [compareId, setCompareId] = useState(null);
  const [stripMode, setStripMode] = useState("timeline"); // "timeline" | "results"
  const [timeline, setTimeline] = useState([]);
  const backRef = useRef(null);
  const dialogRef = useDialogFocus(backRef);

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

    fetchVideoTimeline(item.videoKey, item.frameKey || item.id, 60)
      .then((frames) => {
        if (active) {
          setTimeline(frames || []);
        }
      })
      .catch(() => {
      });

    return () => {
      active = false;
    };
  }, [item?.videoKey, item?.real]);

  useEffect(() => {
    setCompareId(null);
  }, [item?.id]);

  if (!item) return null;

  const strip = stripMode === "timeline" && timeline.length > 0 ? timeline : (results || []);
  const seq = strip;
  const curIdx = seq.findIndex((x) => x.id === item.id || (x.frameKey && x.frameKey === item.frameKey));
  const compareItem = compareId ? (results || []).find((r) => r.id === compareId) : null;
  const atStart = curIdx <= 0;
  const atEnd = curIdx === -1 || curIdx >= seq.length - 1;

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
            {item.videoKey} ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {item.camera}
          </span>
        </div>
        <span className="ws-review-pos">{curIdx >= 0 ? `${curIdx + 1} / ${seq.length}` : "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â"}</span>
        <div className="ws-review-head-actions">
          <button className="ws-btn small" onClick={() => handleStep(-1)} title="Previous (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Ãƒâ€šÃ‚Â)" disabled={atStart}>
            <ArrowLeftOutlined /> Prev
          </button>
          <button className="ws-btn small" onClick={() => handleStep(1)} title="Next (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢)" disabled={atEnd}>
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
                  <span className="ws-review-cmp-label">CURRENT ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {item.frameName}</span>
                  <img src={item.image} alt={item.frameName} />
                </div>
                <div className="ws-review-cmp-item">
                  <span className="ws-review-cmp-label">COMPARE ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {compareItem.frameName}</span>
                  <img src={compareItem.image} alt={compareItem.frameName} />
                </div>
              </div>
            ) : (
              <img className="ws-review-img" src={item.image} alt={item.frameName} />
            )}
            <button className="ws-review-nav prev" onClick={() => handleStep(-1)} title="Previous frame (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Ãƒâ€šÃ‚Â)" disabled={atStart}>
              <ArrowLeftOutlined />
            </button>
            <button className="ws-review-nav next" onClick={() => handleStep(1)} title="Next frame (ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢)" disabled={atEnd}>
              <ArrowRightOutlined />
            </button>
          </div>

          <div className="ws-filmstrip">
            <div className="ws-filmstrip-head">
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <span className="ws-filmstrip-tag">
                  {stripMode === "timeline"
                    ? `Video timeline ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· ${item.videoKey} (${timeline.length} keyframes)`
                    : `Search query results (${results?.length || 0} frames)`}
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
            <div className="ws-review-row"><dt className="k">Timestamp</dt><dd className="v amber">{item.timecode} ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {fmtDur(item.timestamp)}</dd></div>
            <div className="ws-review-row"><dt className="k">Rank</dt><dd className="v">{item.rank != null ? item.rank : "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â"}</dd></div>
            <div className="ws-review-row"><dt className="k">Score</dt><dd className="v">{item.score != null ? `${Math.round(item.score * 100)}%` : "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â"}</dd></div>
            <div className="ws-review-row"><dt className="k">Camera</dt><dd className="v">{item.camera}</dd></div>
            <div className="ws-review-row"><dt className="k">Source</dt><dd className="v">{item.real ? "broadcast feed" : `${item.folderKey} / ${item.videoKey}`}</dd></div>
          </dl>
          <p className="ws-review-note">Score &amp; provenance are placeholders until the backend stabilizes.</p>

          <div className="ws-review-cmp-field">
            <label className="ws-param-label">Compare with another result</label>
            <select value={compareId || ""} onChange={(e) => setCompareId(e.target.value || null)}>
              <option value="">ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â select a frame ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â</option>
              {(results || []).filter((r) => r.id !== item.id).map((r) => (
                <option key={r.id} value={r.id}>
                  #{String(r.rank ?? "ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â")} ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {r.frameName} ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· {r.timecode}
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
