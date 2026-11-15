# AI model

Dịch câu ngắn cho Tày, Ê Đê, Chăm, H'Mông và Jrai. Mặc định dùng bảng câu đã duyệt. Cặp chưa có model trả `model_not_ready`, không bịa câu.

`nllb` dùng NLLB-200 cho Việt ↔ Anh, cộng adapter LoRA cho các cặp dân tộc ↔ Việt đã huấn luyện. Bản dịch máy có tiếng dân tộc luôn kèm cảnh báo chưa rà soát. OCR và ASR trả `unloaded` cho đến khi có checkpoint đã kiểm tra.

Catalog nằm ở `shared/languages.json` (trong repo đủ, hoặc cạnh thư mục này). Có thể trỏ `LANGUAGES_CATALOG`.

```powershell
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --port 8100
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
```

## Huấn luyện

Notebook Kaggle đánh giá ByT5 + LoRA cho Tày → Việt nằm tại
[`notebooks/kaggle/`](notebooks/kaggle/README.md), kèm hướng dẫn chuẩn bị input,
chạy GPU và đưa báo cáo về dự án.

```powershell
python -m pip install -e ".[train]"
python training/import_public.py      # corpus công khai -> data/public/pairs.csv
python training/prepare_base.py       # NLLB-200 -> models/nllb-200-distilled-600M (safetensors)
python training/train_langid.py       # models/langid.json
python training/finetune_nllb.py --epochs 2
python training/evaluate.py --adapter models/nllb-ethnic
```

`finetune_nllb.py` tự gộp `data/reviewed_pairs.csv`, tức file reviewer xuất từ `GET /v1/training/pairs.csv`. Câu cộng đồng đã duyệt vào lượt huấn luyện sau.

Chạy server với mô hình:

```powershell
python -m pip install -e ".[nllb]"
$env:ML_MT_ENGINE='nllb'
$env:ML_MT_ADAPTER='models/nllb-ethnic'
python -m uvicorn app.main:app --port 8100
```

## Giọng nói và ảnh

```powershell
python -m pip install -e ".[train]"
python training/import_speech.py --allow-unknown-license   # audio -> data/speech/manifest.csv
python training/prepare_base.py --name vinai/PhoWhisper-small --kind speech
python training/finetune_whisper.py --task speech_translation --dry-run
```

Audio H'Mông công khai có chữ tiếng Việt, nên `speech_translation` (nghe H'Mông, ra tiếng Việt) là việc làm được trước. Clip chưa duyệt chỉ chạy được với `--allow-unverified`, và checkpoint khi đó bị server từ chối nạp. Khi reviewer đã nghe lại, chạy `import_speech.py --mark-verified` rồi huấn luyện.

```powershell
python -m pip install -e ".[asr,ocr]"
$env:ML_ASR_ENGINE='whisper'
$env:ML_ASR_MODEL='models/whisper-hmn'
$env:ML_OCR_ENGINE='easyocr'
```

Docker: `ML_EXTRAS=nllb,asr,ocr docker compose up --build`. Thư mục `models/` được gắn vào `/models`, nên đường dẫn trong container viết dạng `ML_MT_ADAPTER=/models/nllb-ethnic`. Cổng 8100 chỉ mở cho máy chủ; điện thoại đi qua backend.

Trên Windows, `pyarrow` và `sentencepiece` nạp chung một tiến trình sẽ crash, nên chỉ `import_public.py` đọc parquet.

Nguồn dữ liệu và lý do chọn mô hình nằm trong `RESEARCH.md`. Số đo nằm trong `reports/eval/`.
