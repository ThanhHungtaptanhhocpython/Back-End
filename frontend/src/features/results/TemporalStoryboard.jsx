import { CheckCircleOutlined, PushpinFilled, PushpinOutlined, WarningOutlined } from "@ant-design/icons";
import { fmtDur } from "../../shared/format";

function VerificationBadge({ verification }) {
  if (!verification) return null;
  const decision = verification.vlmDecision;
  if (decision) {
    const tone = decision === "match" ? "ok" : decision === "partial" || decision === "uncertain" ? "warn" : "bad";
    return (
      <span className={`ws-sb-badge ${tone}`} title={verification.vlmReason || ""}>
        VLM {decision}
        {typeof verification.vlmScore === "number" ? ` ${Math.round(verification.vlmScore * 100)}%` : ""}
      </span>
    );
  }
  // No VLM verdict: temporal ordering is valid, but semantic correctness is unverified.
  const status = String(verification.status || "").toLowerCase();
  if (status === "disabled" || status === "" || status === "mock") {
    return (
      <span className="ws-sb-badge warn" title="Ordered sequence from visual retrieval. The semantic sequence verifier was not run.">
        UNVERIFIED
      </span>
    );
  }
  if (status === "fallback") {
    return (
      <span className="ws-sb-badge warn" title="The VLM check failed or timed out; showing the temporal-evidence ranking.">
        VLM FALLBACK
      </span>
    );
  }
  return <span className="ws-sb-badge">{status.toUpperCase()}</span>;
}

export default function TemporalStoryboard({ sequences = [], status, onOpenEvent, onChooseSequence }) {
  if (status === "running") {
    return (
      <div className="ws-storyboard" aria-label="Running temporal search">
        {Array.from({ length: 3 }).map((_, row) => (
          <div className="ws-sb-row skeleton" key={row}>
            <div className="ws-skeleton-line w1" />
            <div className="ws-sb-frames">
              {Array.from({ length: 4 }).map((__, i) => (
                <div className="ws-skeleton-thumb" key={i} />
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!sequences.length) {
    return (
      <div className="ws-empty">
        <WarningOutlined style={{ fontSize: 32, color: "#94a3b8" }} />
        <div>No temporal sequences returned.</div>
        <div className="ws-empty-sub">
          Every event needs a candidate in one shared video, in increasing time order.
        </div>
      </div>
    );
  }

  return (
    <div className="ws-storyboard">
      {sequences.map((sequence) => (
        <div
          className={`ws-sb-row ${sequence.valid ? "" : "invalid"} ${sequence.edited || sequence.chosen ? "edited" : ""}`}
          key={sequence.id}
        >
          <div className="ws-sb-meta">
            <span className="ws-sb-rank">{String(sequence.rank).padStart(2, "0")}</span>
            {onChooseSequence ? (
              <button
                type="button"
                className={`ws-sb-choose ${sequence.chosen ? "on" : ""}`}
                onClick={() => onChooseSequence(sequence, !sequence.chosen)}
                title={
                  sequence.chosen
                    ? "This is the sequence the export will wiggle into ~100 rows. Click to unpick."
                    : "Use this sequence: pin it to rank 1 so Export wiggles it across the 100-row budget."
                }
              >
                {sequence.chosen ? <PushpinFilled /> : <PushpinOutlined />} {sequence.chosen ? "Chosen" : "Use this"}
              </button>
            ) : null}
            <span className="ws-sb-video" title={sequence.videoKey}>
              {sequence.videoKey}
            </span>
            <span className="ws-sb-score">{Math.round((sequence.score || 0) * 100)}%</span>
            {sequence.edited ? (
              <span className="ws-sb-badge ok">
                <CheckCircleOutlined /> EDITED
              </span>
            ) : null}
            <VerificationBadge verification={sequence.verification} />
            {!sequence.valid ? (
              <span className="ws-sb-badge bad" title="Frames must share one video, resolve a frame id, and rise in time">
                <WarningOutlined /> {sequence.orderOk === false ? "OUT OF ORDER" : sequence.sameVideo === false ? "MIXED VIDEO" : "UNRESOLVED"}
              </span>
            ) : null}
          </div>
          <div className="ws-sb-frames">
            {sequence.frames.map((frame) => (
              <button
                type="button"
                className={`ws-sb-frame ${frame.unresolved ? "unresolved" : ""}`}
                key={frame.id}
                onClick={() => onOpenEvent?.(sequence, frame)}
                title={frame.eventQuery || `Event ${frame.eventIndex}`}
              >
                <span className="ws-sb-frame-tag">E{frame.eventIndex}</span>
                {frame.image ? (
                  <img src={frame.image} alt={`Event ${frame.eventIndex}`} loading="lazy" />
                ) : (
                  <div className="ws-sb-frame-missing">no preview</div>
                )}
                <span className="ws-sb-frame-tc">
                  {frame.timecode} · {fmtDur(frame.timestamp)}
                  {frame.eventIndex > 1 ? ` · +${fmtDur(frame.gapSeconds)}` : ""}
                </span>
                <span className="ws-sb-frame-id">
                  #{frame.submissionFrameId ?? "?"}
                  {Object.keys(frame.evidence?.scores || {}).length ? " · evi" : ""}
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
