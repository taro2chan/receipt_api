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
SAVE_OCR_TEXT = True
SAVE_TSV = True
SAVE_DIR = "saved_data"
MODEL_NAME = "gemini-2.0-flash"

# セキュリティトークン（環境変数推奨、デフォルト値は開発用）
SECRET_TOKEN = os.environ.get("MY_APP_TOKEN", "my-secret-key-123")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Geminiの初期設定
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =========================
# データモデル定義
# =========================
class ReceiptItem(BaseModel):
    name: str
    qty: Optional[int] = None
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
# ユーティリティ関数
# =========================
def ensure_dir():
    if (SAVE_OCR_TEXT or SAVE_TSV) and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

def safe_filename(name: str) -> str:
    """ファイル名に使えない文字をアンダースコアに置換"""
    return re.sub(r"[^\w\-ぁ-んァ-ヶ一-龠]", "_", name)

def copy_to_clipboard(text: str):
    """Macのpbcopyを使用してテキストをクリップボードにコピー"""
    try:
        process = subprocess.Popen('pbcopy', stdin=subprocess.PIPE)
        process.communicate(text.encode('utf-8'))
    except Exception as e:
        print(f"Clipboard Error: {e}")

def build_tsv(data: ReceiptData) -> str:
    """ReceiptDataオブジェクトからTSV文字列を生成"""
    lines = []
    # ヘッダー（概要）
    lines.append("\t".join([
        data.datetime or "",
        data.store or "",
        str(data.total_yen or ""),
        str(data.tax_yen or ""),
        data.payment or ""
    ]))
    # 明細
    for item in data.items:
        lines.append("\t".join([
            "", # 日付列を空けて明細であることを示す
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
    """AIを使用してOCRテキストを構造化データに変換"""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    prompt = f"""
OCRテキストからレシート情報を抽出し、必ず以下のJSONスキーマに完全に一致する形で返してください。
リスト([ ])で囲わず、単体オブジェクト({{ }})で出力してください。

スキーマ:
{{
  "store": "店名(string)",
  "datetime": "YYYY-MM-DD HH:MM(string)",
  "total_yen": 合計金額(integer),
  "tax_yen": 消費税額(integer),
  "payment": "支払い方法(string)",
  "items": [
    {{ "name": "名", "qty": 数, "unit_yen": 単価, "line_yen": 小計, "tax_rate": 税率 }}
  ]
}}

OCRテキスト:
{text}
"""
    model = genai.GenerativeModel(MODEL_NAME)
    config = {"temperature": 0, "response_mime_type": "application/json"}
    
    response = model.generate_content(prompt, generation_config=config)
    
    # デバッグ出力
    print(f"--- Gemini Raw Response ---\n{response.text}\n---------------------------")
    
    data = json.loads(response.text)
    # リストで返ってきた場合の救済
    if isinstance(data, list) and data:
        data = data[0]
        
    return ReceiptData.model_validate(data)

def save_output(data: ReceiptData, raw_text: str):
    """結果をファイルとして保存"""
    ensure_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    store_name = safe_filename(data.store or "unknown")
    base_path = os.path.join(SAVE_DIR, f"{store_name}_{ts}")

    if SAVE_OCR_TEXT:
        with open(f"{base_path}_ocr.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)

    if SAVE_TSV:
        with open(f"{base_path}.tsv", "w", encoding="utf-8") as f:
            f.write(build_tsv(data))

# =========================
# FastAPI エンドポイント
# =========================
app = FastAPI()

@app.post("/parse")
async def parse_receipt(request: ReceiptRequest, x_api_token: Optional[str] = Header(None)):
    if x_api_token != SECRET_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Token")

    try:
        parsed_data = call_gemini(request.text)
    except Exception as e:
        print(f"AI Logic Error: {e}")
        parsed_data = ReceiptData(items=[])

    save_output(parsed_data, request.text)
    return build_tsv(parsed_data)

# =========================
# メイン実行ブロック
# =========================
if __name__ == "__main__":
    # 引数がある場合はコマンドラインモード（Terminalから実行）
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        if not os.path.exists(target_file):
            print(f"File not found: {target_file}")
            sys.exit(1)

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"🚀 Processing: {target_file}")
        try:
            result_data = call_gemini(content)
            save_output(result_data, content)
            tsv_text = build_tsv(result_data)
            
            # Mac用クリップボードコピー
            copy_to_clipboard(tsv_text)
            
            print(f"\n--- Result ---\n{tsv_text}")
            print("✅ クリップボードにコピーしました。Excelにペーストできます。")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # 引数がない場合はサーバーモード
    else:
        print(f"📡 Starting FastAPI Server on http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)