"""Generate a local blinded A/B listening page from an answer key.

The HTML shows only slots A/B and the reference text; the condition mapping stays
in the answer-key JSON. Audio is referenced relative to the output page, so the
page and WAVs must stay together.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..errors import InputError


def render_ab_player(
    answer_key: list[dict[str, Any]], *, title: str = "Private blinded A/B"
) -> str:
    if not answer_key:
        raise InputError("answer key has no items")
    cards = []
    for item in answer_key:
        conditions = item.get("conditions") or {}
        if set(conditions) != {"A", "B"}:
            raise InputError(f"item {item.get('item')!r} must have A and B slots")
        audio = []
        for slot in ("A", "B"):
            info = conditions[slot]
            name = Path(str(info["audio_path"])).name
            audio.append(
                f'<div class="slot"><div class="label">Slot {slot}</div>'
                f'<audio controls preload="metadata" src="{html.escape(name)}"></audio>'
                f'<div class="meta">{float(info.get("seconds", 0)):.2f} s</div></div>'
            )
        cards.append(
            '<section class="item">'
            f"<h2>{html.escape(str(item.get('item', '')))} "
            f'<span class="ref">{html.escape(str(item.get("reference", "")))}</span></h2>'
            f'<p class="text">{html.escape(str(item.get("text", "")))}</p>'
            f'<div class="slots">{"".join(audio)}</div>'
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
        padding: 0 1rem; color: #222; }}
h1 {{ font-size: 1.4rem; }}
.item {{ border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin: 1rem 0; }}
h2 {{ font-size: 1rem; margin: 0 0 .4rem; }}
.ref {{ color: #666; font-weight: normal; }}
.text {{ margin: .2rem 0 .8rem; line-height: 1.4; }}
.slots {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.slot {{ flex: 1 1 260px; }}
.label {{ font-weight: 600; margin-bottom: .3rem; }}
audio {{ width: 100%; }}
.meta {{ color: #666; font-size: .85rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Listen to A and B for each prompt. The condition mapping is in the separate
answer-key file; this page does not reveal it. Audio is private evaluation only.</p>
{"".join(cards)}
</body></html>
"""


def write_ab_player(
    answer_key_path: str | Path,
    output_html: str | Path,
    *,
    title: str = "Private blinded A/B",
) -> dict[str, Any]:
    key_path = Path(answer_key_path)
    if not key_path.is_file():
        raise InputError(f"answer key does not exist: {key_path}")
    answer_key = json.loads(key_path.read_text(encoding="utf-8"))
    if not isinstance(answer_key, list):
        raise InputError("answer key root must be a list")
    output = Path(output_html)
    for item in answer_key:
        for info in (item.get("conditions") or {}).values():
            audio = Path(str(info["audio_path"]))
            if not audio.is_file():
                raise InputError(f"audio file is missing: {audio}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_ab_player(answer_key, title=title), encoding="utf-8")
    return {
        "html": output.as_posix(),
        "items": len(answer_key),
        "audio_dir": key_path.parent.as_posix(),
        "answer_key": key_path.as_posix(),
        "note": "Blinded page; conditions remain only in the answer key.",
    }


__all__ = ["render_ab_player", "write_ab_player"]
