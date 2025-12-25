import os
import json
import re
import sys
import subprocess
from datetime import datetime
from typing import List, Optional

import google.generativeai as genai
import uvicorn
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

# =========================
# 設定・定数
# =========================
SAVE_DIR = "saved_data"
MODEL_NAME = "gemini-2.0-flash"

# セキュリティトークン
SECRET_TOKEN = os.environ.get("MY_APP_TOKEN", "my-secret-key-123")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# データモデル
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
        process = subprocess.Popen('pbcopy', stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
    except Exception as e:
        print(f"Clipboard Error: {e}")

def build_tsv(data: ReceiptData) -> str:
    lines = []
    # 概要行（1-5列目）
    lines.append("\t".join([
        data.datetime or "",
        data.store or "",
        str(data.total_yen or ""),
        str(data.tax_yen or ""),
        data.payment or ""
    ]))
    # 明細行（4列分右にずらすため、先頭に5つのタブを入れる）
    for item in data.items:
        lines.append("\t".join([
            "", "", "", "", "", # 概要列分(A-E列)を空ける
            item.name or "",
            str(item.qty or ""),
            str(item.unit_yen or ""),
            str(item.line_yen or ""),
            str(item.tax_rate or "")
        ]))
    return "\n".join(lines) + "\n"

# =========================
# コアロジック（メインエンジン）
# =========================
def call_gemini(text: str) -> ReceiptData:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    # 財産であるプロンプトを復元
    prompt = f"""
OCRテキストからレシート情報を抽出し、必ず以下のJSONスキーマに完全に一致する形で返してください。
リスト([ ])で囲わず、単体オブジェクト({{ }})で出力してください。

【特殊ルール】
- 49...で始まる13桁の数字(JANコード)は無視してください。
- 商品名の前の「＊」や「#s」などの記号は削除してください。
- 割引（▶会員割引など）がある場合、可能であれば最終的な支払額を優先してください。
- 店名はできるだけ正確な名称を抽出してください。

スキーマ:
{{
  "store": "店名(string)",
  "datetime": "YYYY-MM-DD HH:MM(string)",
  "total_yen": 合計金額(integer),
  "tax_yen": 消費税額(integer),
  "payment": "支払い方法(string)",
  "items": [
    {{ "name": "商品名", "qty": 1, "unit_yen": 単価, "line_yen": 小計, "tax_rate": 8または10 }}
  ]
}}

OCRテキスト:
{text}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    config = {"temperature": 0, "response_mime_type": "application/json"}
    response = model.generate_content(prompt, generation_config=config)
    
    print(f"--- Gemini Response ---\n{response.text}\n-----------------------")
    
    data = json.loads(response.text)
    if isinstance(data, list) and data: data = data[0]
    return ReceiptData.model_validate(data)

def process_workflow(ocr_text: str) -> str:
    """共通の処理フロー：一時ファイル作成 -> 解析 -> 昇格 or エラー保存"""
    ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(SAVE_DIR, f"processing_{ts}_ocr.txt")
    
    # 1. 一時ファイルとして保存
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    try:
        # 2. 解析実行
        parsed_data = call_gemini(ocr_text)
        tsv_text = build_tsv(parsed_data)
        
        # 3. 成功：店名で正式保存し、tmpを削除
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
        # 4. 失敗：tmpをunknown_errorにリネームして残す
        error_path = os.path.join(SAVE_DIR, f"unknown_error_{ts}_ocr.txt")
        if os.path.exists(tmp_path):
            os.rename(tmp_path, error_path)
        return f"ERROR: 解析に失敗しました。ファイルを確認してください。\n{e}"

# =========================
# 実行エントリーポイント
# =========================
app = FastAPI()

@app.post("/parse")
async def parse_receipt_api(request: ReceiptRequest, x_api_token: Optional[str] = Header(None)):
    if x_api_token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")
    return process_workflow(request.text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Terminalモード
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
        print(f"📡 Starting FastAPI Server on http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)