"""JSONL I/O utilities and prompt template loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Union


def write_jsonl(path: Union[str, Path], records: Iterable[Any]) -> None:
    """Write an iterable of dicts or Pydantic models to a JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "model_dump"):
                line = json.dumps(record.model_dump(), ensure_ascii=False)
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")


def load_prompt_template(prompts_dir: Union[str, Path], filename: str) -> str:
    """Load a prompt template file from the prompts directory."""
    return (Path(prompts_dir) / filename).read_text(encoding="utf-8")
