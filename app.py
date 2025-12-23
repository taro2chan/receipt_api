# app.py
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

from google import genai


# =========================================================
# 設定（ここだけ触る）
# =========================================================

SAVE_OCR_TEXT = True      # ← デフォルトOFF（必要な期間だけ True にする）
SAVE_MODE = "local"       # "local" or "icloud"

LOCAL_SAVE_DIR = Path.cwd() / "ocr_texts"
ICLOUD_SAVE_DIR = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/ReceiptsOCR"

MODEL_NAME = "models/gemini-2.0-flash"

# =========================================================


app = FastAPI()


# =========================
# Request body
# =========================
class ParseRequest(BaseModel):
    text: str


# =========================
# OCR保存機能（ここに全て閉じ込める）
# =========================
def _get_save_dir() -> Path:
    base = ICLOUD_SAVE_DIR if SAVE_MODE == "icloud" else LOCAL_SAVE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def maybe_save_ocr_text(text: str) -> Optional[Path]:
    """
    OCRテキスト保存はこの関数だけ。
    この関数と呼び出し1行を消せば、機能は完全に外せる。
    """
    if not SAVE_OCR_TEXT:
        return None

    save_dir = _get_save_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_dir / f"ocr_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# =========================
# Utility
# =========================
def _extract_json_block(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    if not (s.startswith("{") and s.endswith("}")):
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            s = m.group(0)
    return s


def _coerce_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        t = x.replace("¥", "").replace(",", "").replace("・", "").replace("·", "")
        t = re.sub(r"[^\d\-]", "", t)
        if t == "" or t == "-":
            return None
        try:
            return int(t)
        except ValueError:
            return None
    return None


def normalize_parsed(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "store": d.get("store"),
        "datetime": d.get("datetime"),
        "total_yen": _coerce_int(d.get("total_yen")),
        "tax_yen": _coerce_int(d.get("tax_yen")),
        "payment": d.get("payment"),
        "items": [],
    }

    for it in d.get("items", []) or []:
        if not isinstance(it, dict):
            continue
        name = it.get("name")
        if not name:
            continue
        out["items"].append({
            "name": name,
            "qty": _coerce_int(it.get("qty")),
            "unit_yen": _coerce_int(it.get("unit_yen")),
            "line_yen": _coerce_int(it.get("line_yen")),
            "tax_rate": _coerce_int(it.get("tax_rate")),
        })

    return out


def to_tsv_multiline(parsed: Dict[str, Any]) -> str:
    lines = []
    header = [
        parsed.get("datetime") or "",
        parsed.get("store") or "",
        str(parsed.get("total_yen") or ""),
        str(parsed.get("tax_yen") or ""),
        parsed.get("payment") or "",
    ]
    lines.append("\t".join(header))

    for it in parsed.get("items", []):
        lines.append("\t".join([
            it.get("name") or "",
            "" if it.get("qty") is None else str(it.get("qty")),
            "" if it.get("unit_yen") is None else str(it.get("unit_yen")),
            "" if it.get("line_yen") is None else str(it.get("line_yen")),
            "" if it.get("tax_rate") is None else str(it.get("tax_rate")),
        ]))

    return "\n".join(lines) + "\n"


# =========================
# Gemini parse（1回呼び）
# =========================
def gemini_parse_receipt(text: str) -> Dict[str, Any]:
    client = genai.Client(api_key=_require_gemini_key())

    prompt = f"""
以下は日本のレシートOCRテキストです。
内容を読み取り、JSONのみで返してください。

JSONスキーマ:
{{
  "store": string|null,
  "datetime": string|null,
  "total_yen": int|null,
  "tax_yen": int|null,
  "payment": string|null,
  "items": [
    {{
      "name": string,
      "qty": int|null,
      "unit_yen": int|null,
      "line_yen": int|null,
      "tax_rate": int|null
    }}
  ]
}}

ルール:
- items には商品行のみ
- 小計/合計/税/注意書きは items に入れない
- 数値は円の整数
- (6個x@128) は qty=6, unit_yen=128, line_yen=768

text:
{text}
""".strip()

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0},
    )

    raw = resp.text if isinstance(resp.text, str) else str(resp)
    data = json.loads(_extract_json_block(raw))
    return normalize_parsed(data)


def _require_gemini_key() -> str:
    import os
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return key


# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/parse")
def parse(req: ParseRequest):
    try:
        # OCRテキスト保存（ONのときだけ）
        maybe_save_ocr_text(req.text)

        parsed = gemini_parse_receipt(req.text)
        tsv = to_tsv_multiline(parsed)
        return PlainTextResponse(tsv)

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"status": "ng", "error": type(e).__name__, "message": str(e)},
        )