"""Save a Hugging Face base model as local safetensors.

Official NLLB-200 and PhoWhisper repos only ship pytorch_model.bin, which
transformers refuses to load on torch < 2.6. The weights are read once with
weights_only=True.

    python training/prepare_base.py
    python training/prepare_base.py --name vinai/PhoWhisper-small --kind speech
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--kind", choices=("text", "speech"), default="text")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "models" / args.name.split("/")[-1]

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        AutoTokenizer,
        GenerationConfig,
    )

    if (output / "model.safetensors").is_file():
        print(f"Đã có {output}")
        return
    model_class = AutoModelForSpeechSeq2Seq if args.kind == "speech" else AutoModelForSeq2SeqLM
    preprocessor = (AutoProcessor if args.kind == "speech" else AutoTokenizer).from_pretrained(args.name)
    config = AutoConfig.from_pretrained(args.name)
    if "model.safetensors" in list_repo_files(args.name):
        model = model_class.from_pretrained(args.name)
    else:
        state = torch.load(hf_hub_download(args.name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        model = model_class.from_config(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.tie_weights()
        tied = ("embed_tokens.weight", "lm_head.weight", "proj_out.weight")
        real_missing = [key for key in missing if not key.endswith(tied)]
        if real_missing or unexpected:
            raise SystemExit(f"Checkpoint lệch kiến trúc: missing={real_missing} unexpected={unexpected}")
    model.generation_config = GenerationConfig.from_pretrained(args.name)
    model.save_pretrained(output, safe_serialization=True)
    preprocessor.save_pretrained(output)
    print(f"Đã lưu {output}")


if __name__ == "__main__":
    main()
