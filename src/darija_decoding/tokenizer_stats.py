"""Tokenize parallel English / Darija / Arabizi text with each model's tokenizer.

Needs only the tokenizers (no model, no GPU). Writes one record per sentence
pair with every token and its character span, for the Tokenizer tab of
tools/token_viewer.html, and prints a summary table.

Usage:
    python -m darija_decoding.tokenizer_stats
    python -m darija_decoding.tokenizer_stats --models gemma4-12b qwen3.5-9b
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

from darija_decoding.prompts import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LANGS = ("english", "darija", "arabizi")
# A word: letters, digits and Arabic diacritics (no punctuation).
WORD = re.compile(r"[^\W_][\wً-ٰٟ]*")


def words_of(text: str) -> list[list]:
    return [[m.group(), m.start(), m.end()] for m in WORD.finditer(text)]


def tokens_of(tokenizer: Tokenizer, text: str) -> list[list]:
    enc = tokenizer.encode(text, add_special_tokens=False)
    pieces = [tokenizer.decode([i]) for i in enc.ids]
    return [[i, p, s, e] for i, p, (s, e) in zip(enc.ids, pieces, enc.offsets)]


def pieces_per_word(words: list[list], tokens: list[list]) -> list[int]:
    """Number of tokens overlapping each word's character span."""
    return [sum(1 for _, _, s, e in tokens if s < we and e > ws) for _, ws, we in words]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/parallel/atlasia_en_darija_sample.jsonl")
    parser.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs/models.yaml")
    parser.add_argument("--models", nargs="+", help="model keys from configs/models.yaml (default: all)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/tokenization")
    args = parser.parse_args()

    models = load_yaml(args.models_config)["models"]
    keys = args.models or list(models)
    tokenizers = {k: Tokenizer.from_pretrained(models[k]["hf_id"]) for k in keys}

    with open(args.data, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    # totals[model][lang] = [words, tokens, single-token words, chars]
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.data.stem}.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for n, pair in enumerate(pairs):
            texts = {lang: pair[lang] for lang in LANGS if pair.get(lang)}
            words = {lang: words_of(t) for lang, t in texts.items()}
            tok = {}
            for key, tokenizer in tokenizers.items():
                tok[key] = {lang: tokens_of(tokenizer, t) for lang, t in texts.items()}
                for lang in texts:
                    ppw = pieces_per_word(words[lang], tok[key][lang])
                    t = totals[key][lang]
                    t[0] += len(ppw)
                    t[1] += len(tok[key][lang])
                    t[2] += sum(1 for c in ppw if c == 1)
                    t[3] += len(texts[lang])
            record = {
                "kind": "tokenization",
                "id": n,
                "row_idx": pair.get("row_idx"),
                "source": pair.get("source"),
                "texts": texts,
                "words": words,
                "tok": tok,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{len(pairs)} pairs -> {out_path}\n")
    print(f"{'tokenizer':<16}{'language':<10}{'words':>8}{'tokens/word':>13}{'1-token words':>15}{'chars/token':>13}")
    for key in keys:
        for lang in LANGS:
            w, t, single, chars = totals[key][lang]
            if w:
                print(f"{key:<16}{lang:<10}{w:>8}{t / w:>13.2f}{single / w:>14.0%}{chars / t:>13.2f}")


if __name__ == "__main__":
    main()
