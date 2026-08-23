/** Deterministic demo implementations for the copilot (chat + translation).
 * These are placeholder implementations, not real AI - see shared/adapters.js. */

const DEMO_DICT_VI_EN = {
  "khung hình": "frame",
  "video": "video",
  "người": "person",
  "xe tải": "truck",
  "hành lý": "luggage",
  "thời gian": "timestamp",
  "máy ảnh": "camera",
  "trạm": "gate",
  "tìm kiếm": "search",
  "kết quả": "results",
};

const DEMO_DICT_EN_VI = {
  "frame": "khung hình",
  "person": "người",
  "truck": "xe tải",
  "luggage": "hành lý",
  "timestamp": "thời gian",
  "camera": "máy ảnh",
  "gate": "trạm",
  "search": "tìm kiếm",
  "results": "kết quả",
};

export function demoTranslate(text, dir) {
  const dict = dir === "vi-en" ? DEMO_DICT_VI_EN : DEMO_DICT_EN_VI;
  let out = String(text || "");
  Object.keys(dict).forEach((k) => {
    out = out.replace(new RegExp(k, "gi"), dict[k]);
  });
  return out;
}

export function mockChatReply(text, frames) {
  const q = String(text || "").trim();
  const f = frames && frames.length ? frames[0] : null;
  const od = f && f.odClasses && f.odClasses.length ? f.odClasses.join(", ") : "-";
  const ctx = f
    ? `Grounded on ${f.videoKey} at ${f.timecode} (rank ${f.rank ?? "-"}). Detected objects: ${od}.`
    : "No frame was attached to this question, so I'm answering in general terms.";
  return `${ctx} DEMO - your question "${q}" was routed to the QA head. Once the real backend is connected, a multimodal answer referencing OCR/object tags will appear here.`;
}
