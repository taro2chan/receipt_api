# app.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Any, Dict, List
import os
import json
import re

from google import genai
from google.genai import types

app = FastAPI()

# =========================
# Config
# =========================
MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")

# =========================
# Request body
# =========================
class ReceiptText(BaseModel):
    text: str

# =========================
# Helpers
# =========================
def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _extract_json_from_text(text_out: str) -> Dict[str, Any]:
    """
    Geminiの出力からJSONだけを安全に取り出す
    """
    t = text_out.strip()

    # ```json ... ``` を剥がす
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # { ... } だけを抜く保険
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)

    return json.loads(t)

def _s(v) -> str:
    return "" if v is None else str(v)

def to_tsv(parsed: dict) -> str:
    """
    TSV:
    1行目: RECEIPT
    2行目以降: ITEM（複数可）
    """
    lines: List[str] = []

    receipt_cols = [
        "RECEIPT",
        _s(parsed.get("datetime")),
        _s(parsed.get("store")),
        _s(parsed.get("total_yen")),
        _s(parsed.get("tax_yen")),
        _s(parsed.get("payment")),
    ]
    lines.append("\t".join(receipt_cols))

    for it in parsed.get("items", []):
        item_cols = [
            "ITEM",
            _s(it.get("name")),
            _s(it.get("qty")),
            _s(it.get("unit_yen")),
            _s(it.get("line_yen")),
            _s(it.get("tax_rate")),
        ]
        lines.append("\t".join(item_cols))

    return "\n".join(lines) + "\n"

# =========================
# Gemini parse (1 call)
# =========================
def gemini_parse_receipt(text: str) -> Dict[str, Any]:
    api_key = _require_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""
あなたは日本のレシートOCRテキストを家計簿入力用に構造化します。
次の text から情報を抽出し、**JSONのみ**で返してください（説明文は禁止）。

出力JSONスキーマ（必須）:
{{
  "store": string|null,
  "datetime": string|null,   // "2025-12-21 16:49"
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
- items には商品だけ入れる（小計・合計・税などは除外）
- 金額は必ず int（円）
- (6個x@128) → qty=6, unit_yen=128, line_yen=768
- 行がずれていても近接行を対応づけてよい
- 推測しすぎない
- JSON以外は絶対に出力しない

text:
\"\"\"{text}\"\"\"
""".strip()

    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    text_out = getattr(resp, "text", None)
    if not isinstance(text_out, str) or not text_out.strip():
        text_out = str(resp)

    return _extract_json_from_text(text_out)

# =========================
# Routes
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "server is running"}

@app.post("/parse")
def parse_receipt(data: ReceiptText):
    try:
        parsed = gemini_parse_receipt(data.text)
        tsv = to_tsv(parsed)
        return PlainTextResponse(tsv, media_type="text/plain; charset=utf-8")
    except Exception as e:
        return {
            "status": "ng",
            "error": type(e).__name__,
            "message": str(e),
        }