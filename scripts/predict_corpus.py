#!/usr/bin/env python
"""Classify every sentence of the corpus with the fine-tuned model.

One JSON file of per-sentence predictions per book, in the same order as the
lines of the source text. Files already present are skipped, so an interrupted
run resumes where it stopped.

    nohup python scripts/predict_corpus.py > predict.log 2>&1 &
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embodiment import config as cfg  # noqa: E402

SUFFIX = "-result.json"


def sentences(path):
    with Path(path).open(encoding="utf8", errors="replace") as f:
        for line in f:
            yield line


def safe_enumerate(gen):
    """Yield predictions, skipping sentences that blow up on the GPU."""
    n = 0
    while True:
        try:
            yield n, next(gen)
        except StopIteration:
            break
        except RuntimeError as err:
            print(f"  skipped sentence {n}: {err}", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            n += 1


def main(model_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("CUDA:", torch.cuda.is_available(), flush=True)

    pipe = pipeline("text-classification", model=str(model_path),
                    tokenizer=AutoTokenizer.from_pretrained(model_path),
                    device=0 if torch.cuda.is_available() else -1,
                    batch_size=64, truncation=True, max_length=512,
                    return_iterator=True)

    files = sorted(f for f in cfg.CORPUS_DIR.iterdir() if f.suffix == ".txt")
    done = sum(1 for f in files if (out_dir / (f.stem + SUFFIX)).exists())
    print(f"{len(files)} files, {done} already done", flush=True)

    for i, path in enumerate(files, start=1):
        result_path = out_dir / (path.stem + SUFFIX)
        if result_path.exists():
            continue
        print(f"[{i}/{len(files)}] {path.name}", flush=True)

        out = pipe(sentences(path))
        results = out if isinstance(out, list) else [
            r for _, r in safe_enumerate(out)]
        try:
            result_path.write_text(json.dumps(results))
        except KeyboardInterrupt:
            result_path.unlink(missing_ok=True)   # never leave a partial file
            raise

    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=cfg.MODEL_DIR / "roberta-nocontext-20260719-1415-best")
    ap.add_argument("--out", type=Path, default=cfg.PREDICTIONS_DIR)
    args = ap.parse_args()
    main(args.model, args.out)
