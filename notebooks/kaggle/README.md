# Đánh giá ByT5 + LoRA trên Kaggle

Nguồn: [lngthnguyn/notebookae644ab63b](https://www.kaggle.com/code/lngthnguyn/notebookae644ab63b).

- `09b_evaluate_byt5_lora_validation.ipynb`: notebook tải từ Kaggle, gồm 21 ô; giữ nguyên code,
  sửa khai báo `nbformat_minor` từ 4 thành 5 để phù hợp với trường `id` của các ô.
- `kernel-metadata.json`: cấu hình Kaggle và dataset đầu vào `lngthnguyn/vietlangai`.

Notebook đánh giá Tày → Việt bằng `google/byt5-small` và adapter LoRA có sẵn.
Notebook không huấn luyện lại và không sử dụng tập test. Đây là quy trình đánh giá
riêng; API của dự án hiện dùng NLLB, chưa hỗ trợ nạp adapter ByT5 này.

## Chạy trên Kaggle

1. Mở notebook nguồn hoặc import file `.ipynb` vào Kaggle.
2. Bật GPU T4/P100 và Internet.
3. Gắn dataset chứa đúng một adapter (`adapter_config.json` và
   `adapter_model.safetensors` hoặc `adapter_model.bin`) cùng `validation.csv`.
   CSV phải có hai cột `source_text`, `target_text`. Có thể dùng ZIP đầu vào từ
   notebook 09A như hướng dẫn trong notebook; notebook 09A không nằm trong bản tải này.
4. Chạy các ô theo thứ tự. Nếu cài thư viện yêu cầu restart kernel, restart rồi chạy lại.
5. Có thể thử `MAX_ROWS=24` trước; dùng `MAX_ROWS=None` để đánh giá toàn bộ validation.
   Mặc định `BEAMS_TO_EVALUATE=(1,)`; đổi thành `(1, 2, 4)` khi cần so sánh beam.
6. Tải `VietLangAI_Kaggle_Validation_Results.zip` ở `/kaggle/working/`,
   hoặc dùng Save Version để lưu output.

Kết quả gồm dự đoán, chrF++, BLEU, exact match, tỷ lệ lặp/rỗng, phân tích theo
độ dài câu và mẫu rà soát. Giải nén báo cáo vào
`reports/eval/byt5-kaggle/<ten-lan-chay>/` nếu cần lưu cùng dự án; giữ
`eval_manifest.json` để đối chiếu cấu hình và checksum.

Cache được ghi mỗi 32 câu. Khi tiếp tục ở phiên Kaggle mới, gắn output cũ làm
Input rồi chép cache và manifest tương ứng vào `EVAL_DIR` trước bước sinh bản dịch.

Notebook ghi dữ liệu có điều kiện CC BY-NC 4.0 và có thể có quyền liên quan từ
nguồn gốc; làm theo hướng dẫn quyền sử dụng trong notebook khi chia sẻ dữ liệu
hoặc báo cáo có chứa câu nguồn và câu dịch.

## Cập nhật code từ Kaggle

Sau khi đăng nhập bằng `kaggle auth login`, chạy từ thư mục gốc dự án:

```powershell
kaggle kernels pull lngthnguyn/notebookae644ab63b -p tmp/kaggle-update -m
```

Lệnh này cập nhật notebook và metadata, không tải dataset, trọng số adapter hay
kết quả thực thi. Các phụ thuộc đánh giá được cài ngay trong notebook.

Sau khi tải, chép notebook vào `notebooks/kaggle/09b_evaluate_byt5_lora_validation.ipynb`,
giữ `nbformat_minor=5` và cập nhật metadata với `code_file` trỏ tới tên file này.
Thư mục `tmp/kaggle-update` chỉ dùng tạm, không đưa vào commit.
