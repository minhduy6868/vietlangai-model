# Nghiên cứu năm ngôn ngữ

Phạm vi đã chốt trong đề cương: Tày, Ê Đê, Chăm, H'Mông, Jrai. Ba ngôn ngữ đánh giá sâu trước là Tày, Ê Đê, Chăm. H'Mông và Jrai có trong sản phẩm, dữ liệu mỏng hơn thì mở rộng sau.

Dịch tiếng Anh không huấn luyện riêng. Câu dân tộc đi tới tiếng Việt, rồi Việt ↔ Anh bằng NLLB vì cặp này đã có sẵn.

## Tày (`tyz`)

- Họ Tai-Kadai, chữ Latin theo Quốc ngữ.
- Corpus công khai: [HeyDunaX/tay-vietnamese-nmt](https://huggingface.co/datasets/HeyDunaX/tay-vietnamese-nmt), 20.600 cặp, 2025.
- Nguyễn Long và cộng sự, workshop AAAI 2026, corpus Tày–Việt và Ba Na kèm mô hình: https://osf.io/fmq5k/ và bài [Not All Data Augmentation Works](https://longstnguyen.com/files/papers/nguyen-et-al-aaai-workshop-2026-not-all-data-augmentation-works.pdf).

## Ê Đê (`rad`)

- Nhóm Chamic, chữ Latin, Tây Nguyên.
- [NIRVLab/rhade-vietnamese-mt](https://huggingface.co/datasets/NIRVLab/rhade-vietnamese-mt), 17.097 cặp. Nguồn có văn bản tôn giáo; phải xem license trước khi huấn luyện.

## Chăm Đông (`cjm`)

- Cùng nhóm Chamic với Ê Đê và Jrai. Chữ truyền thống là Akhar Thrah; học liệu còn bản Latin.
- Có nghiên cứu dịch Việt–Chăm theo mẫu: https://champa.one/article/74. Câu song song neural công khai còn ít.
- Giai đoạn một: chuyển giao từ Ê Đê/Jrai cho bản Latin. OCR Akhar Thrah là việc riêng, không gộp vào OCR Latin.

## H'Mông (`hmn`)

- Họ Hmong-Mien. Chính tả Latin dùng ở Việt Nam khác RPA của cộng đồng hải ngoại.
- Âm thanh + chữ + tiếng Việt từ VOV: [elego/hmongv_audio_speaker03_3k_rows_30042026](https://huggingface.co/datasets/elego/hmongv_audio_speaker03_3k_rows_30042026). Đây là nguồn ASR đầu tiên.
- [Pakorn2112/whisper-model-large-hmong](https://huggingface.co/Pakorn2112/whisper-model-large-hmong) là Hmong–Thái / RPA. Chỉ tham khảo kỹ thuật, không dùng làm mô hình tiếng H'Mông Việt Nam.

## Jrai (`jra`)

- Chamic, gần Ê Đê, chữ Latin, Gia Lai và Kon Tum.
- Đỗ Thị Thuận, Đại học Đà Nẵng, 2012: dịch thống kê Jrai–Việt bằng Moses. Corpus không có trên Hugging Face.
- Từ điển đối chiếu khoảng 613 mục: https://glottosphere.com/dictionary/jra.
- [21uyennt/jarai_vie_bible](https://huggingface.co/datasets/21uyennt/jarai_vie_bible): 30.979 câu Kinh Thánh Việt (bản 1951) – Jrai (bản 1973). Không ghi license; `import_public.py` chỉ nhận khi có `--allow-unknown-license`.
- Giai đoạn một dùng chuyển giao từ Ê Đê, giữ chính tả Jrai khi nhập câu mới.

## Nguồn tham khảo thêm (tháng 11/2026)

- CKTN, arXiv [2607.08362](https://arxiv.org/abs/2607.08362): 44.367 văn bản đơn ngữ Chăm, Khmer, Tày-Nùng. Dùng cho tiền huấn luyện tiếp, không phải câu song song.
- H'Mông có audio Việt Nam: [ThaiKami/hmong-vn-audio](https://huggingface.co/datasets/ThaiKami/hmong-vn-audio) (278 đoạn), [nhatminh/hmong_low_bleu](https://huggingface.co/datasets/nhatminh/hmong_low_bleu) (1.884 đoạn). Chưa ghi license, chưa kiểm chính tả trong nước.
- ASR tiếng Việt làm điểm xuất phát: [PhoWhisper](https://github.com/VinAIResearch/PhoWhisper), [PhoASR](https://github.com/Qualcomm-AI-research/PhoASR).
- Cách làm cộng đồng: [Masakhane](https://github.com/masakhane-io/masakhane-mt), người bản ngữ cùng thu và duyệt dữ liệu. Ứng với luồng đóng góp → reviewer → `reviewed_pairs.csv` → huấn luyện lại.

## Mô hình

1. Dịch chữ: `facebook/nllb-200-distilled-600M` + một adapter LoRA cho mọi cặp dân tộc ↔ Việt, học cả hai chiều. Script: `training/finetune_nllb.py`.
   - Đo trên 3.000 câu mỗi ngôn ngữ: NLLB tách tiếng Việt 1,34 token/từ, Tày 1,73, không mất ký tự Tày. mT5 tách tiếng Việt 2,2 token/từ. ViT5 mất 9,5% token Ê Đê và 13% token Jrai.
   - NLLB không có `ƀ Ƀ Ŏ Ĭ Ŭ`. `app/textnorm.py` đổi `ƀ/Ƀ` sang `ɓ/Ɓ` (chữ Hausa, không có trong năm ngôn ngữ) và tách `Ŏ Ĭ Ŭ` thành chữ + dấu trăng, rồi đổi ngược ở đầu ra.
   - Mỗi ngôn ngữ có thẻ riêng (`tyz_Latn`, `rad_Latn`, …), khởi tạo bằng trung bình thẻ họ hàng: Aceh (`ace_Latn`, Chamic duy nhất trong NLLB) cho Ê Đê, Jrai, Chăm; Việt, Thái, Lào cho Tày.
   - Việt ↔ Anh dùng mô hình gốc, tắt adapter. Một mô hình trong bộ nhớ phục vụ cả đường Tày → Việt → Anh.
   - LoRA r=32 chạy trên GPU 4 GB. `--full` để fine-tune toàn bộ trên GPU từ 16 GB (Kaggle, Colab).
2. `mt5-small` (`training/finetune_mt5.py`) giữ làm mốc so sánh.
3. Nhận diện ngôn ngữ: Naive Bayes trên n-gram ký tự 1–4 (`training/train_langid.py`), chỉ trả lời khi tin cậy từ 0,9. Chữ Chăm vẫn nhận theo bảng mã; tiếng Anh theo từ vựng.
4. Giọng nói: PhoWhisper-small (`training/finetune_whisper.py`). Kiểm 278 đoạn ThaiKami thì cả 278 có chữ là tiếng Việt, không phải chữ H'Mông; `nhatminh/hmong_low_bleu` lệch độ dài giữa audio và chữ. Vì vậy có hai việc riêng:
   - `speech_translation`: nghe H'Mông, viết tiếng Việt. Làm được ngay với dữ liệu đang có.
   - `asr`: nghe H'Mông, viết chữ H'Mông trong nước. Chờ bản ghi đã duyệt.
   Checkpoint chỉ được nạp khi có WER trên tập test và `trained_on_unverified=false`. Dữ liệu công khai chưa ghi license nên hiện chỉ chạy thử.
5. Ảnh chữ Latin trên điện thoại dùng ML Kit. Server dự phòng bằng EasyOCR tiếng Việt (`ML_OCR_ENGINE=easyocr`), kèm cảnh báo `ƀ ĕ ŏ ŭ` có thể sai.

## Lộ trình

1. Dịch chữ Tày, Ê Đê ↔ Việt bằng NLLB + LoRA; nhận diện ngôn ngữ. Số đo ở `reports/eval/`.
2. Jrai: nhóm xác nhận quyền dùng bộ Kinh Thánh hoặc thu câu đời thường qua màn đóng góp, rồi chạy lại `finetune_nllb.py`. Ê Đê và Jrai cùng nhánh nên học chung một adapter.
3. Chăm Latin và H'Mông: chưa có câu song song công khai. Thu qua cộng đồng, reviewer duyệt, `reviewed_pairs.csv` tự vào lượt huấn luyện sau.
4. Giọng H'Mông: nhóm xác nhận quyền dùng audio ThaiKami, reviewer nghe lại câu tiếng Việt, xuất manifest với `--mark-verified`, rồi huấn luyện `speech_translation`. ASR chữ H'Mông cần thu bản ghi chính tả trong nước. Đo WER trên người nói không có trong train.
5. Tiền huấn luyện tiếp trên CKTN (Chăm, Tày-Nùng đơn ngữ) trước khi fine-tune, nếu điểm Tày/Chăm chững lại.
6. OCR Akhar Thrah: mô hình riêng, không gộp OCR Latin.

## Không đưa vào năm ngôn ngữ này

Khmer có corpus VOV lớn và NLLB (`khm_Khmr`), bài SOICT 2024 của HUST. Ba Na có corpus EAAI 2024 khoảng 20.000 cặp. Cơ Ho có ViKhoMT. Mường có `mr4/trans-vi-mu` nhưng dữ liệu không công khai. Giữ làm phương án mở rộng, không thay năm ngôn ngữ đã chốt.
