from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Any, Dict
import os
import json
import re
import time
from pathlib import Path

import google.generativeai as genai

app = FastAPI()

# ====== Request body ======
class ReceiptText(BaseModel):
    text: str


# ====== Model ======
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


# ====== Daily limit (safety) ======
# 例: export DAILY_LIMIT=50
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "50"))

# カウンタ保存先（プロジェクト直下に .usage.json を作る）
USAGE_FILE = Path(os.environ.get("USAGE_FILE", ".usage.json"))

def _today_str() -> str:
    return time.strftime("%Y-%m-%d")

def _load_usage() -> Dict[str, Any]:
    if not USAGE_FILE.exists():
        return {"day": _today_str(), "count": 0}
    try:
        return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        # 壊れてたら今日0から
        return {"day": _today_str(), "count": 0}

def _save_usage(usage: Dict[str, Any]) -> None:
    # 原子的に書く（壊れにくくする）
    tmp = USAGE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(usage, ensure_ascii=False), encoding="utf-8")
    tmp.replace(USAGE_FILE)

def check_daily_limit() -> None:
    usage = _load_usage()
    today = _today_str()

    if usage.get("day") != today:
        usage = {"day": today, "count": 0}

    count = int(usage.get("count", 0))
    if count >= DAILY_LIMIT:
        raise RuntimeError(f"Daily limit exceeded: {DAILY_LIMIT}")

    usage["count"] = count + 1
    _save_usage(usage)


# ====== Gemini helpers ======
def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _extract_text_from_gemini_response(resp: Any) -> str:
    # 1) resp.text
    if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
        return resp.text.strip()

    # 2) candidates[0].content.parts[0].text
    try:
        cands = getattr(resp, "candidates", None)
        if cands and len(cands) > 0:
            content = getattr(cands[0], "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts and len(parts) > 0:
                    t = getattr(parts[0], "text", None)
                    if isinstance(t, str) and t.strip():
                        return t.strip()
    except Exception:
        pass

    # 3) dict response
    if isinstance(resp, dict):
        for path in [
            ("text",),
            ("candidates", 0, "content", "parts", 0, "text"),
            ("candidates", 0, "output"),
        ]:
            cur: Any = resp
            ok = True
            for key in path:
                if isinstance(key, int):
                    if isinstance(cur, list) and len(cur) > key:
                        cur = cur[key]
                    else:
                        ok = False
                        break
                else:
                    if isinstance(cur, dict) and key in cur:
                        cur = cur[key]
                    else:
                        ok = False
                        break
            if ok and isinstance(cur, str) and cur.strip():
                return cur.strip()

    # 4) fallback: grab {...}
    s = str(resp)
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        return m.group(0).strip()

    raise RuntimeError("Could not extract text from Gemini response")

def _coerce_json(text_out: str) -> Dict[str, Any]:
    t = text_out.strip()
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)

    return json.loads(t)

def gemini_parse_receipt(text: str) -> Dict[str, Any]:
    api_key = _require_env("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    prompt = f"""
あなたはレシートOCRテキストを家計簿入力用に構造化します。
次の text から情報を抽出して、JSONのみで返してください（説明文禁止）。

出力JSONのスキーマ（必ずこのキーを出す）:
{{
  "store": string|null,
  "datetime": string|null,   // 例: "2025-12-21 16:49"
  "total_yen": int|null,
  "tax_yen": int|null,
  "payment": string|null,    // 例: "カード" "現金"
  "items": [
    {{
      "name": string,
      "qty": int|null,
      "unit_yen": int|null,
      "line_yen": int|null,
      "tax_rate": int|null    // 8 or 10。分からなければ null
    }}
  ]
}}

ルール:
- "items" には「商品」だけ入れる。次は items に入れない:
  小計, 合計, 税, 対象額, お預り, お釣り, ポイント, 登録番号, 電話番号, 取引番号, 担当者, 注意書き
- 金額は「¥」「・」「,」などが混ざっても int にする（例: "·1,531" → 1531）
- (6個x@128) のような表記があれば qty=6, unit_yen=128, line_yen=768 を優先
- 1行に品名と金額が揃っていない場合は、近い行同士を対応づけて推測してよい
- 可能なら items の合計が total_yen と矛盾しないようにする（完全一致できなくてもOK）

text:
{text}
""".strip()

    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": 0},
    )

    text_out = _extract_text_from_gemini_response(resp)
    return _coerce_json(text_out)


# ====== TSV output ======
def to_tsv(parsed: dict) -> str:
    cols = [
        parsed.get("datetime") or "",
        parsed.get("store") or "",
        str(parsed.get("total_yen") or ""),
        str(parsed.get("tax_yen") or ""),
        parsed.get("payment") or "",
    ]
    return "\t".join(cols) + "\n"


@app.get("/")
def health():
    return {"status": "ok", "message": "server is running"}


@app.post("/parse")
def parse_receipt(data: ReceiptText):
    try:
        # ここで「1日N回まで」を強制
        check_daily_limit()

        parsed = gemini_parse_receipt(data.text)
        return PlainTextResponse(to_tsv(parsed))
    except Exception as e:
        # ショートカット運用を考えて、テキストで返す
        msg = f"ERROR\t{type(e).__name__}\t{str(e)}\n"
        return PlainTextResponse(msg, status_code=500)