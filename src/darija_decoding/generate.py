"""Generate answers for a prompt set with one model (steps 1b and 1c).

Each answer is appended to outputs/generations/<model>__<prompt set>.jsonl
as soon as it is generated, with every generated token and its top-k
alternatives. Re-running the same command skips answers that are already
saved, so an interrupted run picks up where it stopped.

Usage:
    python -m darija_decoding.generate --model gemma4-12b --dry-run
    python -m darija_decoding.generate --model gemma4-12b --limit 2
    python -m darija_decoding.generate --model gemma4-12b
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from darija_decoding.prompts import Prompt, build_prompts, load_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Thinking blocks some chat models emit even with thinking turned off.
THINKING_PATTERNS = [
    re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL),  # Gemma 4
    re.compile(r"<think>.*?</think>", re.DOTALL),  # Qwen
]


def load_model(hf_id: str, quant_cfg: dict):
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])
    quantization = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    kwargs = dict(quantization_config=quantization, dtype=compute_dtype, device_map="auto")

    processor = AutoProcessor.from_pretrained(hf_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except ValueError:
        # Some multimodal checkpoints only register a multimodal class.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(hf_id, **kwargs)
    model.eval()
    return processor, model


def clean_answer(raw: str, tokenizer) -> str:
    """Remove thinking blocks and leftover special tokens from decoded output."""
    text = raw
    for pattern in THINKING_PATTERNS:
        text = pattern.sub("", text)
    for token in tokenizer.all_special_tokens:
        text = text.replace(token, "")
    return text.strip()


def token_details(logits: torch.Tensor, new_ids: torch.Tensor, tokenizer, k: int) -> list[dict]:
    """Per generated token: its probability, rank, the step's entropy and the top-k alternatives.

    `logits` are the raw model outputs (steps x vocab), before temperature or
    top_p, so the numbers mean the same thing for greedy and sampled runs.
    """
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    probs = logprobs.exp()
    chosen = probs.gather(1, new_ids[:, None]).squeeze(1)
    ranks = (probs > chosen[:, None]).sum(dim=-1) + 1
    entropy = -(probs * logprobs).nansum(dim=-1)
    top_probs, top_ids = probs.topk(k, dim=-1)

    def texts(ids: list[int]) -> list[str]:
        return tokenizer.batch_decode([[i] for i in ids])

    chosen_ids = new_ids.tolist()
    chosen_texts = texts(chosen_ids)
    details = []
    for step, token_id in enumerate(chosen_ids):
        alt_ids = top_ids[step].tolist()
        details.append(
            {
                "id": token_id,
                "text": chosen_texts[step],
                "prob": round(chosen[step].item(), 5),
                "rank": ranks[step].item(),
                "entropy": round(entropy[step].item(), 4),
                "top": [
                    [i, t, round(p, 5)] for i, t, p in zip(alt_ids, texts(alt_ids), top_probs[step].tolist())
                ],
            }
        )
    return details


def generate_one(processor, model, prompt: Prompt, gen_cfg: dict, setting: dict) -> dict:
    tokenizer = getattr(processor, "tokenizer", processor)

    messages = []
    if gen_cfg.get("system_prompt"):
        messages.append({"role": "system", "content": gen_cfg["system_prompt"]})
    messages.append({"role": "user", "content": prompt.text})

    chat = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=gen_cfg.get("enable_thinking", False),
    )
    # The chat template already contains BOS and other special tokens.
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if setting.get("do_sample"):
        transformers.set_seed(gen_cfg["seed"])

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=gen_cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,  # raw logits, before temperature/top_p
            **setting,
        )
    seconds = time.perf_counter() - start

    new_ids = output.sequences[0, input_len:]
    logits = torch.stack(output.logits, dim=1)[0]
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    return {
        "chat_input": chat,
        "raw_output": raw,
        "answer": clean_answer(raw, tokenizer),
        "n_new_tokens": len(new_ids),
        "hit_max_tokens": len(new_ids) >= gen_cfg["max_new_tokens"],
        "seconds": round(seconds, 2),
        # Every generated token, including any thinking-block tokens at the start.
        "tokens": token_details(logits, new_ids, tokenizer, gen_cfg["top_k_logged"]),
    }


def saved_keys(path: Path) -> set[tuple[str, str, str, str]]:
    """Answers already in the output file. Includes the prompt text hash, so
    an answer to an older wording of a prompt doesn't count as done."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys.add((r["prompt_id"], r["version"], r["setting"], r.get("prompt_hash")))
    return keys


def write_run_meta(path: Path, args, hf_id: str, quant_cfg: dict, gen_cfg: dict, model) -> None:
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "hf_id": hf_id,
        "prompts_file": str(args.prompts.relative_to(REPO_ROOT)),
        "quantization": quant_cfg,
        "generation": gen_cfg,
        # Model defaults (e.g. top_k, repetition_penalty) apply unless overridden.
        "model_generation_config": model.generation_config.to_dict(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model key from configs/models.yaml")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/prompts/exploration_v1.yaml")
    parser.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs/models.yaml")
    parser.add_argument("--gen-config", type=Path, default=REPO_ROOT / "configs/generation.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/generations")
    parser.add_argument("--limit", type=int, help="only the first N prompt ids (all their versions)")
    parser.add_argument("--versions", nargs="+", help="only these versions, e.g. darija_ar msa")
    parser.add_argument("--settings", nargs="+", help="only these decoding settings, e.g. greedy")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and exit without loading a model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_cfg = load_yaml(args.models_config)
    gen_cfg = load_yaml(args.gen_config)

    if args.model not in models_cfg["models"]:
        raise SystemExit(f"Unknown model {args.model!r}. Choose from: {', '.join(models_cfg['models'])}")
    hf_id = models_cfg["models"][args.model]["hf_id"]

    prompts = build_prompts(args.prompts, args.versions)
    if args.limit:
        keep = list(dict.fromkeys(p.prompt_id for p in prompts))[: args.limit]
        prompts = [p for p in prompts if p.prompt_id in keep]

    settings = {name: s for name, s in gen_cfg["settings"].items() if not args.settings or name in args.settings}
    jobs = [(p, name) for p in prompts for name in settings]

    out_path = args.out_dir / f"{args.model}__{args.prompts.stem}.jsonl"
    done = saved_keys(out_path)
    todo = [(p, name) for p, name in jobs if (p.prompt_id, p.version, name, p.text_hash) not in done]
    print(f"{len(jobs)} generations planned, {len(jobs) - len(todo)} already saved, {len(todo)} to run")

    if args.dry_run:
        for p, name in todo:
            print(f"\n--- {p.prompt_id} | {p.version} | {name}\n{p.text}")
        return
    if not todo:
        return

    print(f"Loading {hf_id} ...")
    processor, model = load_model(hf_id, models_cfg["quantization"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(out_path.with_suffix(".meta.json"), args, hf_id, models_cfg["quantization"], gen_cfg, model)

    with open(out_path, "a", encoding="utf-8") as out:
        for i, (prompt, name) in enumerate(todo, 1):
            result = generate_one(processor, model, prompt, gen_cfg, settings[name])
            record = {
                "model": args.model,
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "version": prompt.version,
                "setting": name,
                "prompt": prompt.text,
                "prompt_hash": prompt.text_hash,
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{i}/{len(todo)}] {prompt.prompt_id} {prompt.version} {name}: "
                f"{result['n_new_tokens']} tokens in {result['seconds']}s"
            )
            print(f"    {result['answer'][:120]!r}")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
