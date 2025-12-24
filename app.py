from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os
import json
import re
from datetime import datetime
import google.generativeai as genai

# =========================
# 設定スイッチ（ここだけ）
# =========================
SAVE_OCR_TEXT = True   # True にすると OCR を保存
SAVE_TSV = True        # True にすると TSV を保存
SAVE_DIR = "saved_data" # 保存先ディレクトリ

MODEL_NAME = "gemini-2.0-flash"

# =========================

app = FastAPI()

class ReceiptText(BaseModel):
    text: str


def ensure_dir():
    if (SAVE_OCR_TEXT or SAVE_TSV) and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```json", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("JSON not found in LLM output")
    return json.loads(m.group(0))


def call_llm(text: str) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    prompt = f"""
次のOCRテキストを日本のレシートとして解析し、JSONのみで返してください。

スキーマ:
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
- 分からない項目は null
- 推測はしてよいが無理はしない
- JSON以外は出力しない

OCR:
{text}
"""

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(prompt, generation_config={"temperature": 0})

    out = resp.text if hasattr(resp, "text") else str(resp)
    return extract_json_from_text(out)


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-ぁ-んァ-ヶ一-龠]", "_", name)


def save_files(parsed: Dict[str, Any], ocr_text: str):
    ensure_dir()

    dt_raw = parsed.get("datetime")
    try:
        dt = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M")
    except Exception:
        dt = datetime.now()

    ts = dt.strftime("%Y%m%d_%H%M")
    store = parsed.get("store") or "unknown"
    store = safe_filename(store)

    if SAVE_OCR_TEXT:
        with open(os.path.join(SAVE_DIR, f"{store}_{ts}_ocr.txt"), "w", encoding="utf-8") as f:
            f.write(ocr_text)

    if SAVE_TSV:
        tsv = build_tsv(parsed)
        with open(os.path.join(SAVE_DIR, f"{store}_{ts}.tsv"), "w", encoding="utf-8") as f:
            f.write(tsv)


def build_tsv(parsed: Dict[str, Any]) -> str:
    lines = []
    lines.append("\t".join([
        parsed.get("datetime") or "",
        parsed.get("store") or "",
        str(parsed.get("total_yen") or ""),
        str(parsed.get("tax_yen") or ""),
        parsed.get("payment") or ""
    ]))

    for item in parsed.get("items", []):
        lines.append("\t".join([
            "",
            item.get("name") or "",
            str(item.get("qty") or ""),
            str(item.get("unit_yen") or ""),
            str(item.get("line_yen") or ""),
            str(item.get("tax_rate") or "")
        ]))

    return "\n".join(lines) + "\n"


@app.post("/parse")
def parse_receipt(data: ReceiptText):
    parsed = {}
    try:
        parsed = call_llm(data.text)
    except Exception:
        parsed = {
            "store": None,
            "datetime": None,
            "total_yen": None,
            "tax_yen": None,
            "payment": None,
            "items": []
        }
    finally:
        save_files(parsed, data.text)

    return build_tsv(parsed)