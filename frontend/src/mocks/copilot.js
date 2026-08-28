/** Deterministic fallback implementations for the copilot (chat + translation).
 * These keep the UI usable when backend providers are offline. */

const DEMO_DICT_VI_EN = [
  ["Đoạn clip cần tìm là cảnh", "The clip to find shows"],
  ["đoạn clip cần tìm", "the clip to find"],
  ["hai người phụ nữ", "two women"],
  ["một người", "one person"],
  ["người kia", "the other person"],
  ["phụ nữ", "woman"],
  ["người", "person"],
  ["đang cho dê ăn", "feeding goats"],
  ["cho dê ăn", "feeding goats"],
  ["dê", "goats"],
  ["trong chuồng", "in a pen"],
  ["chuồng trại", "farm pen"],
  ["chuồng", "pen"],
  ["không gian trại rộng rãi", "a spacious farm area"],
  ["không gian", "space"],
  ["trại", "farm"],
  ["rộng rãi", "spacious"],
  ["mái che bằng tôn", "a corrugated metal roof"],
  ["mái che", "roof shelter"],
  ["hàng rào gỗ", "wooden fences"],
  ["chia thành nhiều dãy chuồng", "divided into several rows of pens"],
  ["dài dê được nuôi nhốt", "where goats are kept"],
  ["được nuôi nhốt", "are kept"],
  ["mặc áo thun trắng", "wearing a white T-shirt"],
  ["áo thun trắng", "white T-shirt"],
  ["quàng áo đỏ trên vai", "with a red garment over the shoulder"],
  ["áo đỏ", "red garment"],
  ["mặc áo dài tay kẻ sọc tím truyền thống", "wearing a traditional purple striped long-sleeved shirt"],
  ["áo dài tay", "long-sleeved shirt"],
  ["kẻ sọc tím", "purple striped"],
  ["truyền thống", "traditional"],
  ["cả hai", "both"],
  ["đều mỉm cười", "are smiling"],
  ["mỉm cười", "smiling"],
  ["tỏ vẻ thích thú", "look interested"],
  ["khung hình", "frame"],
  ["hình ảnh", "image"],
  ["tìm kiếm", "search"],
  ["kết quả", "results"],
  ["thời gian", "timestamp"],
];

const DEMO_DICT_EN_VI = [
  ["the clip to find shows", "đoạn clip cần tìm là cảnh"],
  ["two women", "hai người phụ nữ"],
  ["feeding goats", "cho dê ăn"],
  ["goats", "dê"],
  ["in a pen", "trong chuồng"],
  ["farm pen", "chuồng trại"],
  ["wooden fences", "hàng rào gỗ"],
  ["white T-shirt", "áo thun trắng"],
  ["red garment", "áo đỏ"],
  ["purple striped", "kẻ sọc tím"],
  ["long-sleeved shirt", "áo dài tay"],
  ["traditional", "truyền thống"],
  ["smiling", "mỉm cười"],
  ["frame", "khung hình"],
  ["person", "người"],
  ["search", "tìm kiếm"],
  ["results", "kết quả"],
  ["timestamp", "thời gian"],
];

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function applyDictionary(text, entries) {
  let out = String(text || "");
  entries
    .slice()
    .sort((a, b) => b[0].length - a[0].length)
    .forEach(([source, target]) => {
      out = out.replace(new RegExp(escapeRegExp(source), "gi"), target);
    });
  return out;
}

export function demoTranslate(text, dir) {
  const input = String(text || "").trim();
  if (!input) return "";
  return applyDictionary(input, dir === "vi-en" ? DEMO_DICT_VI_EN : DEMO_DICT_EN_VI);
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
