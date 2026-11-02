"""Download a random sample of an English-Darija parallel dataset.

Uses the Hugging Face dataset viewer API, so only the sampled rows are
downloaded (the full atlasia dataset is ~820 MB). Rows are picked at random
positions, so each source appears roughly in proportion to its size.

Usage:
    python -m darija_decoding.sample_parallel
    python -m darija_decoding.sample_parallel --n-requests 100 --rows-per-request 50
"""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API = "https://datasets-server.huggingface.co"
COLUMNS = ("english", "darija", "arabizi", "source", "script_type")


def get_json(path: str, params: dict, retries: int = 6) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except Exception as err:
            if attempt == retries - 1:
                raise
            # The API rate-limits unauthenticated clients: back off hard on 429.
            wait = 15 * (attempt + 1) if "429" in str(err) else 3 * (attempt + 1)
            print(f"  {err}; retrying in {wait}s")
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="atlasia/english-to-darija-arabic-script-formatted")
    parser.add_argument("--n-requests", type=int, default=60, help="random positions to read from")
    parser.add_argument("--rows-per-request", type=int, default=80, help="consecutive rows read at each position (max 100)")
    parser.add_argument("--pause", type=float, default=1.5, help="seconds between requests, to stay under the rate limit")
    parser.add_argument("--max-chars", type=int, default=400, help="skip pairs whose Darija is longer (whole documents)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/parallel/atlasia_en_darija_sample.jsonl")
    args = parser.parse_args()

    base = {"dataset": args.dataset, "config": "default", "split": "train"}
    total = get_json("rows", {**base, "offset": 0, "length": 1})["num_rows_total"]
    rng = random.Random(args.seed)
    offsets = sorted(rng.sample(range(total - args.rows_per_request), args.n_requests))
    print(f"{args.dataset}: {total} rows, reading {args.n_requests} x {args.rows_per_request}")

    pairs, seen = [], set()
    for k, offset in enumerate(offsets, 1):
        rows = get_json("rows", {**base, "offset": offset, "length": args.rows_per_request})["rows"]
        for r in rows:
            row = {c: r["row"].get(c) for c in COLUMNS}
            if not row["english"] or not row["darija"] or len(row["darija"]) > args.max_chars:
                continue
            if r["row_idx"] in seen:
                continue
            seen.add(r["row_idx"])
            pairs.append({"row_idx": r["row_idx"], **row})
        if k % 10 == 0:
            print(f"  {k}/{len(offsets)} requests, {len(pairs)} pairs kept")
        time.sleep(args.pause)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_arabizi = sum(1 for p in pairs if p["arabizi"])
    print(f"Saved {len(pairs)} pairs ({n_arabizi} with Arabizi) to {args.out}")


if __name__ == "__main__":
    main()
