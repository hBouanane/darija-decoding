"""Decoding interventions.

Word map: for each MSA word and its Darija replacement, find the first token
where their spellings diverge, and lower the MSA token there. The Darija
token is boosted only when the two words share pieces before that point
(e.g. " كيف" + "اش" for كيفاش): the boost then fires only right after the
shared piece. A Darija boost with no shared piece would fire at every word
start, which in practice pushed words like " ملي" or " شي" into sentences
that had no MSA at all.
This uses transformers' `sequence_bias`: a bias on a sequence's last token,
applied only when the tokens before it match.

Inspect what a word map does to a tokenizer (no model needed):
    python -m darija_decoding.interventions --model gemma4-12b
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from darija_decoding.prompts import load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# How a word can appear: after a space or at line start, with or without و ("and").
WORD_FORMS = (" {}", "{}", " و{}", "و{}")


def load_word_map(path: Path) -> list[tuple[str, list[str], float | None]]:
    """(msa, darija words, optional per-pair bias) for each entry."""
    return [(p["msa"], p["darija"], p.get("bias")) for p in load_yaml(path)["pairs"]]


def word_map_bias(
    encode: Callable[[str], list[int]], pairs: list[tuple[str, list[str], float | None]], bias: float
) -> tuple[dict[tuple[int, ...], float], list[dict]]:
    """Build a `sequence_bias` dict, plus a report of what each pair became.

    `bias` is the default strength; a pair's own `bias` overrides it. A key
    with no shared prefix (a single token) applies everywhere, so the MSA
    side is only lowered that way when its token is the whole word.
    """
    seq_bias: dict[tuple[int, ...], float] = {}
    report = []

    def put(key: tuple[int, ...], value: float) -> None:
        if key in seq_bias and (seq_bias[key] > 0) != (value > 0):
            del seq_bias[key]  # one pair boosts what another lowers: leave it alone
        else:
            seq_bias[key] = value

    for msa, darija_words, pair_bias in pairs:
        strength = bias if pair_bias is None else pair_bias
        for form in WORD_FORMS:
            m_ids = encode(form.format(msa))
            for darija in darija_words:
                d_ids = encode(form.format(darija))
                k = 0
                while k < min(len(m_ids), len(d_ids)) and m_ids[k] == d_ids[k]:
                    k += 1
                boost = 0 < k < len(d_ids)
                lower = k < len(m_ids) and (k > 0 or len(m_ids) == 1)
                if boost:
                    put(tuple(d_ids[: k + 1]), strength)
                if lower:
                    put(tuple(m_ids[: k + 1]), -strength)
                report.append(
                    {"msa": msa, "darija": darija, "form": form.format(""), "shared": k, "bias": strength,
                     "msa_ids": m_ids, "darija_ids": d_ids, "boost": boost, "lower": lower}
                )
    return seq_bias, report


def main() -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemma4-12b", help="model key from configs/models.yaml")
    parser.add_argument("--word-map", type=Path, default=REPO_ROOT / "data/wordlists/msa_to_darija.yaml")
    args = parser.parse_args()

    hf_id = load_yaml(REPO_ROOT / "configs/models.yaml")["models"][args.model]["hf_id"]
    tok = Tokenizer.from_pretrained(hf_id)
    encode = lambda s: tok.encode(s, add_special_tokens=False).ids
    parser_default_bias = 3.0  # same default as generate.py --word-bias
    seq_bias, report = word_map_bias(encode, load_word_map(args.word_map), parser_default_bias)

    def pieces(ids, k):
        return " + ".join(("[" if i == k else "") + tok.decode([t]) + ("]" if i == k else "") for i, t in enumerate(ids))

    print(f"{len(seq_bias)} biased token sequences. [x] = the token where MSA and Darija diverge.\n")
    for r in report:
        if r["form"] != " ":
            continue
        status = "ok" if r["boost"] and r["lower"] else ("boost only" if r["boost"] else ("lower only" if r["lower"] else "SKIPPED"))
        print(f"{r['msa']:>8} -> {r['darija']:<8} bias {r['bias']:<4g} shared {r['shared']}  {status:<11} "
              f"MSA: {pieces(r['msa_ids'], r['shared'])}   Darija: {pieces(r['darija_ids'], r['shared'])}")


if __name__ == "__main__":
    main()
