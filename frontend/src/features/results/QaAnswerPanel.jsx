function numeric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function QaAnswerPanel({ tab }) {
  if (tab?.searchType !== "QA" || tab?.status !== "done") return null;

  const summary = tab?.meta || {};
  const itemAnswer = tab?.results?.find((item) => String(item?.answer || "").trim())?.answer;
  const answer = String(summary.answer || itemAnswer || "").trim();
  if (!answer) return null;

  const isDemo = Boolean(summary.demo) || tab?.resultSource === "demo" || tab?.resultSource === "fallback";
  const confidence = numeric(summary.confidence);
  const evaluatedFrames = numeric(summary.evaluated_frames);
  const retrievedFrames = numeric(summary.retrieved_frames);
  const answerMode = String(summary.answer_mode || "").toLowerCase();
  const status = answerMode === "best_guess"
    ? "PHƯƠNG ÁN KHẢ DĨ NHẤT"
    : answerMode === "fallback"
      ? "KHÔNG ĐỦ BẰNG CHỨNG"
      : "ĐÃ KIỂM CHỨNG";

  return (
    <section className={`ws-qa-answer ${isDemo ? "demo" : "live"}`} aria-label="QA generated answer" aria-live="polite">
      <div className="ws-qa-answer-head">
        <div>
          <div className="ws-qa-answer-eyebrow">CÂU TRẢ LỜI QA</div>
          <div className="ws-qa-answer-badges">
            <span className={`ws-qa-answer-badge ${isDemo ? "demo" : "live"}`}>{isDemo ? "DỮ LIỆU THỬ" : "VLM TRỰC TIẾP"}</span>
            <span className="ws-qa-answer-status">{status}</span>
          </div>
        </div>
        <div className="ws-qa-answer-stats">
          {confidence !== null ? <span>ĐỘ TIN CẬY <strong>{Math.round(confidence * 100)}%</strong></span> : null}
          {evaluatedFrames !== null ? <span>ĐÃ XEM <strong>{evaluatedFrames}</strong></span> : null}
          {retrievedFrames !== null ? <span>TRUY XUẤT <strong>{retrievedFrames}</strong></span> : null}
        </div>
      </div>

      <p className="ws-qa-answer-text">{answer}</p>
      {summary.reason ? <p className="ws-qa-answer-reason">{summary.reason}</p> : null}
      {isDemo ? (
        <p className="ws-qa-answer-note">
          Test data only. This tab switches to the live VLM automatically when matching local keyframe images are available.
        </p>
      ) : null}
    </section>
  );
}
