"""JSONL I/O utilities and prompt template loading."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Set, Union

# Canonical prompts dir, used as the final fallback when resolving includes (so
# a prompt staged in a temp dir can still pull in shared fragments from here).
_PROMPTS_DIR_DEFAULT = Path(__file__).parent.parent / "prompts"

# {{include: some_file.txt}} — inline another prompt fragment. Lets shared rules
# live in ONE file that every prompt variant references, so editing the rule once
# applies everywhere (no per-variant edits, no Python changes).
_INCLUDE_RE = re.compile(r"\{\{\s*include:\s*([^}]+?)\s*\}\}")


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


def read_jsonl(path: Union[str, Path]) -> list:
    """Read a JSONL file into a list of dicts. Returns ``[]`` if the file is missing."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt_template(prompts_dir: Union[str, Path], filename: str) -> str:
    """Load a prompt template, expanding any ``{{include: file.txt}}`` directives.

    Includes are resolved relative to the including file's directory, then the
    given ``prompts_dir``, then the canonical prompts dir. This lets every prompt
    variant share one rules fragment (edit the rule once, applies everywhere).
    """
    base = Path(prompts_dir)
    return _render_prompt(base / filename, base, set())


def _render_prompt(path: Path, base: Path, seen: Set[str]) -> str:
    text = Path(path).read_text(encoding="utf-8")

    def _replace(match: "re.Match") -> str:
        name = match.group(1).strip()
        if name in seen:
            return ""  # cycle guard
        seen.add(name)
        for cand in (Path(path).parent / name, base / name, _PROMPTS_DIR_DEFAULT / name):
            if cand.is_file():
                return _render_prompt(cand, base, seen).rstrip("\n")
        raise FileNotFoundError(
            f"prompt include not found: '{name}' (referenced by {path})"
        )

    return _INCLUDE_RE.sub(_replace, text)
