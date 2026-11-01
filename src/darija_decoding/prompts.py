"""Load prompt sets and expand them into the concrete prompts sent to a model."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    category: str
    version: str  # darija_ar, darija_arabizi, msa, en, fr
    text: str  # exactly what the model receives as the user message


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompts(path: Path, versions: list[str] | None = None) -> list[Prompt]:
    """Expand every prompt into one Prompt per version.

    Versions with an entry under `instructions` (msa, en, fr) get that
    "answer in Darija" instruction appended on a new line.
    """
    data = load_yaml(path)
    instructions = data.get("instructions", {})

    prompts = []
    for entry in data["prompts"]:
        for version, text in entry["text"].items():
            if versions and version not in versions:
                continue
            instruction = instructions.get(version)
            if instruction:
                text = f"{text}\n{instruction}"
            prompts.append(Prompt(entry["id"], entry["category"], version, text))
    return prompts
