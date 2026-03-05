import pandas as pd
import ollama
import json
from datetime import datetime, timedelta

# --- 設定區 ---
INPUT_FILE = "old_news.csv" 
OUTPUT_FILE = "LLM_score.csv"
MODEL_NAME = "llama3.1:latest"
STOCK_ID = "2330"

def analyze_news(stock_id, title, content=None):
    """ 
    送交 Ollama 分析。
    如果 content 為空或 None，則進行純標題分析。
    """
    if content and str(content).strip():
        mode_desc = f"【新聞標題】：{title}\n    【新聞內文】：{content}"
        evidence_hint = "請從內文摘錄一段具體事實或數據"
    else:
        mode_desc = f"【新聞標題】：{title}\n    （無內文，請僅就標題進行推論）"
        evidence_hint = "標題推論"

    prompt = f"""
    你是一位專業台股分析師。請分析以下新聞對股票代號 {stock_id} 的影響。
    
    {mode_desc}

    請嚴格以 JSON 格式回傳，不要有解釋文字。
    指標定義：
    - sentiment_score: -1.0 (極大利空) 至 1.0 (極大利多)，0.0 為中立。
    - impact_intensity: 0.0 至 1.0 (影響強度)。
    - certainty: 0.0 (傳聞) 至 1.0 (已發生事實)。
    - time_horizon: 根據影響長度僅限填入：0.8 (短期), 0.4 (中期), 0.1 (長期)。

    回傳格式：
    {{
        "sentiment_score": float,
        "impact_intensity": float,
        "certainty": float,
        "time_horizon": float,
        "summary": "一句話重點",
        "evidence": "{evidence_hint}",
        "reason": "評分理由"
    }}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], format='json')
        res_content = response['message']['content'].strip()
        
        # --- 清洗邏輯：移除 Markdown 標籤 ---
        if res_content.startswith("```json"):
            res_content = res_content[7:]
        if res_content.endswith("```"):
            res_content = res_content[:-3]
        res_content = res_content.strip()
        
        return json.loads(res_content)
    except Exception as e:
        print(f"   ❌ LLM 分析出錯: {e}")
        return None

# --- 主程式執行 ---
try:
    # 讀取 CSV
    df_raw = pd.read_csv(INPUT_FILE)
    
    # --- 篩選 20 天內的新聞 ---
    # 支援 'date' 或 'publish_date' 欄位名稱
    date_col = 'date' if 'date' in df_raw.columns else 'publish_date'
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    cutoff_date = datetime.now() - timedelta(days=20)
    df_all = df_raw[df_raw[date_col] >= cutoff_date].copy()
    # -----------------------

    total_count = len(df_all)
    print(f"🚀 開始處理資料（篩選後共 {total_count} 則近 20 天新聞）...")

    results = []
    for index, row in df_all.iterrows():
        print(f"\n👉 處理中: {str(row['title'])[:25]}...")
        
        # 檢查是否有 text 欄位且是否有內容
        news_content = row.get('text') if 'text' in row else None
        
        # 執行分析 (會自動判定是標題+內文還是純標題)
        analysis = analyze_news(STOCK_ID, row['title'], news_content)
        
        if analysis:
            combined_data = {**row.to_dict(), **analysis}
            results.append(combined_data)
            print(f"   ✅ 分析完成！情緒: {analysis.get('sentiment_score')} | 時效: {analysis.get('time_horizon')}")
        else:
            results.append(row.to_dict())

    # 儲存結果
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✨ 任務完成！結果已存至 {OUTPUT_FILE}")

except Exception as e:
    print(f"💥 程式執行失敗: {e}")