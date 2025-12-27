import os
import json
import re
import sys
import subprocess
import yaml
from datetime import datetime
from typing import List, Optional

# 新SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: pip install google-genai pyyaml fastapi uvicorn pydantic")
    sys.exit(1)

import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# =========================
# 設定・定数
# =========================
SAVE_DIR = "saved_data"
MODEL_NAME = "gemini-2.0-flash"
PROMPT_FILE = "prompts.yaml"

SECRET_TOKEN = os.environ.get("MY_APP_TOKEN", "my-secret-key-123")
API_KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None

# =========================
# データモデル (厳密に保持)
# =========================
class ReceiptItem(BaseModel):
    name: str
    qty: Optional[int] = 1
    unit_yen: Optional[int] = None
    line_yen: Optional[int] = None
    tax_rate: Optional[int] = None

class ReceiptData(BaseModel):
    store: Optional[str] = None
    datetime: Optional[str] = None
    total_yen: Optional[int] = None
    tax_yen: Optional[int] = None
    payment: Optional[str] = None
    items: List[ReceiptItem] = []

class ReceiptRequest(BaseModel):
    text: str

# =========================
# ユーティリティ
# =========================
def ensure_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    if not name: return "unknown"
    return re.sub(r"[^\w\-ぁ-んァ-ヶ一-龠]", "_", name)

def copy_to_clipboard(text: str):
    try:
        # macOS想定
        process = subprocess.Popen('pbcopy', stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
    except Exception as e:
        print(f"Clipboard Error: {e}")

def build_tsv(data: ReceiptData) -> str:
    lines = []
    # 概要行
    lines.append("\t".join([
        data.datetime or "",
        data.store or "",
        str(data.total_yen or ""),
        str(data.tax_yen or ""),
        data.payment or ""
    ]))
    # 明細行
    for item in data.items:
        lines.append("\t".join([
            "", "", "", "", "", # A-E列空け
            item.name or "",
            str(item.qty or ""),
            str(item.unit_yen or ""),
            str(item.line_yen or ""),
            str(item.tax_rate or "")
        ]))
    return "\n".join(lines) + "\n"

# =========================
# コアロジック
# =========================
def call_gemini(text: str) -> ReceiptData:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    # YAMLからプロンプト構成を読み込み
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        conf = yaml.safe_load(f)['receipt_task']

    rules_str = "\n".join([f"- {r}" for r in conf['rules']])
    
    # 結合 (f-stringを使わずreplaceで注入することで、スキーマ内の { } を保護)
    prompt = (
        f"{conf['system_instruction']}\n\n"
        f"【特殊ルール】\n{rules_str}\n\n"
        f"スキーマ:\n{conf['json_schema']}\n\n"
        f"{conf['user_template']}"
    ).replace("[[text]]", text)

    # 実行
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )
    )
    
    resp_text = response.text.strip()
    print(f"--- Gemini Response ---\n{resp_text}\n-----------------------")
    
    # JSONパースとバリデーション
    data = json.loads(resp_text)
    if isinstance(data, list) and data:
        data = data[0]
    
    return ReceiptData.model_validate(data)

def process_workflow(ocr_text: str) -> str:
    ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(SAVE_DIR, f"processing_{ts}_ocr.txt")
    
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    try:
        parsed_data = call_gemini(ocr_text)
        tsv_text = build_tsv(parsed_data)
        
        # 成功時の保存
        store_name = safe_filename(parsed_data.store)
        base_path = os.path.join(SAVE_DIR, f"{store_name}_{ts}")
        
        with open(f"{base_path}_ocr.txt", "w", encoding="utf-8") as f:
            f.write(ocr_text)
        with open(f"{base_path}.tsv", "w", encoding="utf-8") as f:
            f.write(tsv_text)
            
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return tsv_text

    except Exception as e:
        print(f"❌ 解析失敗: {e}")
        error_path = os.path.join(SAVE_DIR, f"unknown_error_{ts}_ocr.txt")
        if os.path.exists(tmp_path):
            os.rename(tmp_path, error_path)
        return f"ERROR: 解析に失敗しました。\n{e}"

# =========================
# エントリーポイント
# =========================
app = FastAPI()

@app.post("/parse")
async def parse_receipt_api(request: ReceiptRequest, x_api_token: Optional[str] = Header(None)):
    if x_api_token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return process_workflow(request.text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLIモード
        target = sys.argv[1]
        if not os.path.exists(target):
            print("File not found.")
            sys.exit(1)
        with open(target, "r", encoding="utf-8") as f:
            ocr_content = f.read()
        
        print(f"🚀 Processing: {target}")
        result = process_workflow(ocr_content)
        
        if not result.startswith("ERROR"):
            copy_to_clipboard(result)
            print(f"\n--- Result ---\n{result}\n✅ クリップボードにコピーしました。")
        else:
            print(result)
    else:
        # サーバーモード
        uvicorn.run(app, host="127.0.0.1", port=8000)