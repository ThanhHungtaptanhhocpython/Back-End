function numberOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function candidateFrames(candidate, frames) {
  const names = new Set([
    ...(Array.isArray(candidate?.supporting_frame_names) ? candidate.supporting_frame_names : []),
    candidate?.representative_frame_name,
  ].filter(Boolean).map(String));
  const all = Array.isArray(frames) ? frames : [];
  let matches = all.filter((frame) => names.has(String(frame?.frameName || "")));
  if (!matches.length) {
    matches = all.filter((frame) => String(frame?.videoKey || "") === String(candidate?.video_id || ""));
  }
  return matches.slice(0, 3);
}

export default function QaCandidateList({
  candidates,
  frames,
  selectedId = "",
  onSelect,
  compact = false,
}) {
  const options = Array.isArray(candidates) ? candidates.filter((candidate) => candidate?.answer) : [];
  if (options.length < 2) return null;

  return (
    <div className={`ws-qa-candidates ${compact ? "compact" : ""}`}>
      <div className="ws-qa-candidates-title">PHƯƠNG ÁN THEO TỪNG VIDEO</div>
      <div className="ws-qa-candidates-note">
        Mỗi phương án chỉ dùng ảnh, OCR và lời thoại thuộc cùng một video.
      </div>
      <div className="ws-qa-candidate-grid">
        {options.map((candidate) => {
          const previewFrames = candidateFrames(candidate, frames);
          const confidence = numberOrNull(candidate.confidence);
          const coverage = numberOrNull(candidate.event_coverage);
          const total = numberOrNull(candidate.event_total);
          const selected = String(selectedId) === String(candidate.candidate_id);
          return (
            <article
              key={candidate.candidate_id || candidate.video_id}
              className={`ws-qa-candidate ${selected ? "selected" : ""}`}
            >
              {previewFrames.length ? (
                <div className="ws-qa-candidate-images">
                  {previewFrames.map((frame) => (
                    <img key={frame.id} src={frame.image} alt={frame.frameName || candidate.video_id} />
                  ))}
                </div>
              ) : null}
              <div className="ws-qa-candidate-meta">
                <strong>{candidate.video_id || candidate.candidate_id}</strong>
                {total !== null && total > 0 ? <span>ĐỦ SỰ KIỆN {coverage || 0}/{total}</span> : null}
                {confidence !== null ? <span>TIN CẬY {Math.round(confidence * 100)}%</span> : null}
              </div>
              <div className="ws-qa-candidate-answer">{candidate.answer}</div>
              {candidate.reason ? <div className="ws-qa-candidate-reason">{candidate.reason}</div> : null}
              {typeof onSelect === "function" ? (
                <button
                  type="button"
                  className="ws-qa-candidate-select"
                  onClick={() => onSelect(candidate)}
                  disabled={selected}
                >
                  {selected ? "Đang chọn" : "Chọn phương án này"}
                </button>
              ) : null}
            </article>
          );
        })}
      </div>
    </div>
  );
}
