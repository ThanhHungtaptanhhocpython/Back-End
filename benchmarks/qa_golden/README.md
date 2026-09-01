# Grounded Video Q&A Golden Dataset

Bộ golden này đo riêng pipeline **retrieve → evidence → answer → verify** của tab Q&A.
Nó dùng frame thật trong bộ AIC đã tải, không cần train lại model và không sao chép ảnh vào Git.

## Thành phần

| Dataset | Số câu | Mục tiêu |
|---|---:|---|
| `abstention_v1.jsonl` | 6 | từ chối đoán khi hình không đủ bằng chứng |
| `visual_core_v1.jsonl` | 12 | vật thể, hành động, thiết bị, phương tiện |
| `count_ocr_v1.jsonl` | 10 | đếm và đọc chữ/logo/số trên hình |
| `visual_attributes_v1.jsonl` | 10 | màu sắc và hướng không gian |
| `temporal_multievent_v1.jsonl` | 2 | chuỗi nhiều sự kiện và chữ số rất nhỏ trên màn hình |

Tổng cộng có 40 câu, cân bằng 20 câu tiếng Việt và 20 câu tiếng Anh. Mỗi câu đã được
đối chiếu trực quan với ít nhất một frame thật. Đây là bộ khởi đầu do một người duyệt; trước khi
dùng làm benchmark chính thức, nên có thêm một người duyệt độc lập các đáp án/alias.

`language` là ngôn ngữ của **câu hỏi**. Mọi đáp án tự nhiên trong golden đều phải là
tiếng Việt có dấu, kể cả khi câu hỏi bằng tiếng Anh. Riêng OCR, logo, tên riêng, mã và
con số được giữ nguyên đúng nội dung nhìn thấy thay vì dịch máy.

## Kiểm tra nhanh

Từ thư mục `Back-End`:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts\evaluate_qa_golden.py validate --check-files
```

Lệnh trên kiểm tra schema, ID trùng, giới hạn 100 ký tự và mọi `frame_path` bên dưới
`KEYFRAMES_ROOT`. Với cấu hình hiện tại, kết quả phải là 40 case và 46 evidence frame.

Khi backend đang chạy ở cổng 3000, chạy thử vài câu trước:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts\evaluate_qa_golden.py run --limit 3 --check-files
```

Chạy một câu xác định để debug:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts\evaluate_qa_golden.py run --id vc_vi_001 --top-k 100
```

Chạy toàn bộ:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts\evaluate_qa_golden.py run --check-files
```

Prediction và báo cáo được lưu mặc định trong `benchmarks/qa_runs/` (đã gitignore). Có thể
chấm lại một file mà không gọi model:

```powershell
& .\.venv\Scripts\python.exe -X utf8 scripts\evaluate_qa_golden.py score benchmarks\qa_runs\qa_predictions-YYYYMMDD-HHMMSS.jsonl
```

## Các chỉ số chính

- `answer_accuracy`: đúng đáp án hoặc alias sau chuẩn hóa dấu/cách viết; câu đếm hiểu cả
  `hai` và `2`.
- `answer_language_compliance`: đáp án tự nhiên tuân thủ tiếng Việt; OCR/tên riêng/số
  được coi là nội dung trung tính ngôn ngữ.
- `retrieval_recall_at_K`: frame golden xuất hiện trong K kết quả đầu.
- `video_recall_at_K`: ít nhất lấy đúng video, dù chưa đúng frame.
- `supporting_evidence_accuracy`: model có trỏ một `qa_supporting` đúng frame golden.
- `status_accuracy`: đúng `answered`/`uncertain`.
- `format_compliance`: đáp án không xuống dòng và không quá 100 ký tự.
- `confidence_brier`: độ hiệu chuẩn confidence; càng thấp càng tốt.
- `failure_counts`: phân tách lỗi retrieval, answer, status, format và supporting evidence.

Nhờ tách `retrieval_miss` khỏi `wrong_answer`, ta biết nên cải tiến encoder/index hay prompt/VLM.

## Schema một case

```json
{
  "schema_version": "1.0",
  "id": "vc_vi_001",
  "question": "...",
  "language": "vi",
  "answer_type": "object",
  "expected_status": "answered",
  "answer": "xe máy",
  "aliases": ["mô tô"],
  "evidence": [{
    "split": "L21_a",
    "video_id": "L21_V001",
    "frame_id": "020406",
    "frame_path": "L21_a/L21_V001/020406.webp",
    "timestamp": 680.2,
    "tolerance_seconds": 30.0
  }],
  "difficulty": "easy",
  "tags": ["visual", "vehicle"],
  "review": {
    "status": "manually_verified",
    "reviewer": "...",
    "date": "YYYY-MM-DD"
  }
}
```

## Thêm case mới an toàn

1. Chọn câu hỏi đại diện cho lỗi thật, không chọn vì hệ thống hiện tại đã trả lời được.
2. Mở frame, tự xác nhận đáp án; không lấy chính output của model làm nhãn.
3. Ghi đáp án ngắn nhất và chỉ thêm alias có cùng nghĩa rõ ràng.
   Đáp án tự nhiên phải là tiếng Việt có dấu; không thêm alias tiếng Anh chỉ vì câu hỏi
   được viết bằng tiếng Anh.
4. Dùng đường dẫn tương đối dưới `KEYFRAMES_ROOT`; không di chuyển/sửa ảnh nguồn. Chỉ thêm
   `tolerance_seconds` khi đã xác nhận các frame lân cận vẫn thuộc cùng một sự kiện.
5. Dùng ID mới, tăng version file khi thay đổi ý nghĩa nhãn, rồi chạy `validate --check-files`.
6. Nhờ người thứ hai duyệt các case khó/không rõ trước khi dùng để quyết định model tốt hơn.

Không đưa API key, output model hoặc dữ liệu máy-specific vào các file golden.
