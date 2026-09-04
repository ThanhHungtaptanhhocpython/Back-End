import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { CloseOutlined, CompassOutlined } from "@ant-design/icons";
import useDialogFocus from "../../hooks/useDialogFocus";

/**
 * In-app guidance for choosing a search mode. The two cards at the top answer
 * the question people actually ask -- "Q&A hay Agent Search?" -- and the table
 * plus the short list below place the other modes around them.
 */

const PRIMARY = [
  {
    tag: "Q&A",
    headline: "Bạn có một câu hỏi cần câu trả lời",
    how: 'Chọn tab loại "Q&A", gõ câu hỏi, Enter (tự chạy sau ~0.4s).',
    examples: [
      "Người đứng cạnh xe cứu hỏa mặc áo màu gì?",
      "Biển số chiếc xe tải màu xanh là gì?",
      "Có bao nhiêu người trong cảnh phỏng vấn?",
    ],
    result:
      "Một câu trả lời tiếng Việt + độ tin cậy + lý do, kèm các frame làm bằng chứng. " +
      "Nếu có nhiều video ứng viên, bạn chọn đáp án đúng ở panel trên đầu kết quả.",
    engine:
      "Pipeline cố định: gom bằng chứng visual + OCR + ASR → gửi frame cho VLM sinh câu trả lời theo schema.",
  },
  {
    tag: "Agent Search",
    headline: "Bạn có một mô tả cảnh, muốn AI tự lập kế hoạch tìm",
    how: 'Gõ mô tả vào ô query của tab bất kỳ, bấm nút "Agent Search". Kết quả mở ở tab mới.',
    examples: [
      "Góc quay sát mặt đường, người áo vàng băng qua đường lúc trời mưa",
      "Phóng viên cầm micro phỏng vấn trước tòa nhà, phía sau có logo đài",
      "Người kéo lưới cá, sau đó lên thuyền, rồi thuyền rời bến",
    ],
    result:
      "Danh sách frame đã xếp hạng (một cảnh) HOẶC các chuỗi sự kiện theo thời gian (nếu là temporal). " +
      'Ô "answer" chỉ là tóm tắt AI đã làm gì (query mở rộng, routing, verify) — không phải đáp án nội dung.',
    engine:
      'Có LLM planner + phân loại ý định: câu có "lần lượt / sau đó / rồi đến / theo thứ tự" sẽ tự route sang TRAKE (chuỗi cùng video, timestamp tăng dần).',
  },
];

const COMPARE = [
  ["Mục tiêu", "Trả lời một câu hỏi về nội dung video", "Truy hồi frame/chuỗi khớp một mô tả"],
  ["Input", "Câu hỏi: cái gì / ở đâu / màu gì / bao nhiêu / ai / nói gì", "Mô tả cảnh, càng nhiều chi tiết (màu, góc máy, hành động) càng tốt"],
  ["Nhận về", "Câu trả lời (tiếng Việt) + frame dẫn chứng", "Frame xếp hạng, hoặc chuỗi sự kiện; kèm plan của AI"],
  ["LLM", "Chỉ ở bước sinh câu trả lời (VLM)", "Có planner tách & mở rộng truy vấn + verify"],
  ["Dùng khi", "Đề bài hỏi một giá trị cụ thể", "Mô tả dài/nhiều sự kiện, muốn AI tự tối ưu query"],
  ["Đừng dùng khi", "Chỉ cần quét frame, không cần câu chữ → Text/Agent", "Cần một đáp án → Q&A; hoặc chỉ 1–2 từ khóa → Text"],
];

const OTHER_MODES = [
  ["Text", "Tìm keyframe theo mô tả ngắn (KIS cơ bản). Tự chạy khi gõ, nhanh nhất, không LLM."],
  ["OCR Text", "Tìm theo chữ xuất hiện trong khung hình: biển hiệu, phụ đề, tít báo."],
  ["Image", 'Pivot từ một frame tham chiếu (upload ảnh, hoặc "Similar" trên một kết quả).'],
  ["Temporal", "Bạn tự nhập từng sự kiện theo thứ tự → TRAKE beam search. Agent Search làm tự động bước tách sự kiện này."],
];

const TIPS = [
  "Q&A luôn trả lời tiếng Việt; Agent trả về frame/chuỗi, không phải câu chữ.",
  "Cùng một đề: chạy Q&A để lấy đáp án, rồi Text/Agent để quét thêm frame nộp.",
  'Nút "Agent Search" mờ đi khi ô query trống — phải có mô tả trước.',
  "Cả hai dùng chung backend retrieval đang bật (Jina CLIP v2 / BEiT3).",
];

export default function SearchGuide({ open, onClose }) {
  const closeRef = useRef(null);
  const dialogRef = useDialogFocus(closeRef, open);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onClose]);

  if (!open) return null;

  // Portalled up to `.ws-root` (falling back to <body>) so the fixed backdrop
  // escapes the search sidebar's sticky-positioned stacking context -- without
  // that, the panel behind it, the selection tray and the chat pane paint on
  // top. `.ws-root` (not <body>) keeps the --ws-* design tokens in scope, and
  // puts this at the same level as the other workstation overlays.
  const mount = (typeof document !== "undefined" && document.querySelector(".ws-root")) || document.body;

  return createPortal(
    <div className="ws-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div
        ref={dialogRef}
        className="ws-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Chọn chế độ tìm kiếm"
        tabIndex={-1}
        style={{ width: "min(860px, 96vw)" }}
      >
        <div className="ws-modal-head">
          <div className="ws-modal-title">
            <CompassOutlined /> Chọn chế độ tìm kiếm phù hợp
          </div>
          <button ref={closeRef} className="ws-modal-close" onClick={onClose} title="Đóng (Esc)">
            <CloseOutlined />
          </button>
        </div>

        <div className="ws-modal-body">
          <p className="ws-guide-lead">
            Hai chế độ hay bị nhầm là <b>Q&amp;A</b> và <b>Agent Search</b>. Khác biệt cốt lõi:
            Q&amp;A <i>trả lời một câu hỏi</i>, Agent Search <i>đi tìm frame theo một mô tả</i>.
          </p>

          <div className="ws-guide-cards">
            {PRIMARY.map((mode) => (
              <div key={mode.tag} className="ws-guide-card">
                <div className="ws-guide-card-tag">{mode.tag}</div>
                <div className="ws-guide-card-headline">{mode.headline}</div>

                <div className="ws-guide-field">
                  <span className="ws-guide-field-label">Cách chạy</span>
                  <span>{mode.how}</span>
                </div>

                <div className="ws-guide-field">
                  <span className="ws-guide-field-label">Ví dụ</span>
                  <ul className="ws-guide-examples">
                    {mode.examples.map((ex) => <li key={ex}>{ex}</li>)}
                  </ul>
                </div>

                <div className="ws-guide-field">
                  <span className="ws-guide-field-label">Nhận về</span>
                  <span>{mode.result}</span>
                </div>

                <div className="ws-guide-field ws-guide-field-muted">
                  <span className="ws-guide-field-label">Bên dưới</span>
                  <span>{mode.engine}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="ws-guide-section-title">So sánh nhanh</div>
          <div className="ws-guide-table-wrap">
            <table className="ws-guide-table">
              <thead>
                <tr>
                  <th aria-label="Tiêu chí" />
                  <th>Q&amp;A</th>
                  <th>Agent Search</th>
                </tr>
              </thead>
              <tbody>
                {COMPARE.map(([label, qa, agent]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    <td>{qa}</td>
                    <td>{agent}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="ws-guide-section-title">Các chế độ còn lại</div>
          <div className="ws-guide-other">
            {OTHER_MODES.map(([name, desc]) => (
              <div key={name} className="ws-guide-other-row">
                <span className="ws-guide-other-name">{name}</span>
                <span>{desc}</span>
              </div>
            ))}
          </div>

          <div className="ws-guide-section-title">Mẹo</div>
          <ul className="ws-guide-tips">
            {TIPS.map((tip) => <li key={tip}>{tip}</li>)}
          </ul>
        </div>
      </div>
    </div>,
    mount,
  );
}
