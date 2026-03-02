import pandas as pd
import ollama
import json

# --- 設定區 ---
# 修改為你上傳的檔案名稱
INPUT_FILE = "news_台積電_2026-02-23_to_2026-02-24.csv" 
OUTPUT_FILE = "LLM_score.csv"
MODEL_NAME = "llama3.1:latest"
# 固定分析對象為台積電
STOCK_ID = "2330"

def analyze_full_news(stock_id, title, content):
    """ 送交 Ollama 分析  """
    context_text = content if content else "（無內文，請僅就標題分析）"
    
    prompt = f"""
    你是一位專業台股分析師。請分析以下新聞對股票代號 {stock_id} 的影響。
    
    【新聞標題】：{title}
    【新聞內文】：{context_text}

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
        "evidence": "請從內文摘錄一段具體事實或數據，若無內文則填'標題推論'",
        "reason": "評分理由"
    }}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"   ❌ LLM 分析出錯: {e}")
        return None

# --- 主程式執行 ---
try:
    # 讀取 CSV
    df_all = pd.read_csv(INPUT_FILE)
    total_count = len(df_all)
    print(f"🚀 開始全量處理資料，共 {total_count} 則新聞 ...")

    results = []
    for index, row in df_all.iterrows():
        # 對應 CSV 欄位: title, text
        print(f"\n👉 [{index+1}/{total_count}] 處理中: {str(row['title'])[:20]}...")
        
        # 提取內文
        full_text = str(row['text']) if pd.notna(row['text']) else ""
        
        # 執行分析
        analysis = analyze_full_news(STOCK_ID, row['title'], full_text)
        
        if analysis:
            # 合併原始 CSV 內容與 LLM 分析結果
            combined_data = {**row.to_dict(), **analysis}
            results.append(combined_data)
            print(f"   ✅ 分析完成！情緒: {analysis.get('sentiment_score')} | 時效: {analysis.get('time_horizon')}")
        else:
            # 若分析失敗，保留原始資料並補空值
            results.append(row.to_dict())

    # 儲存結果
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✨ 任務完成！結果已存至 {OUTPUT_FILE}")

except Exception as e:
    print(f"💥 程式執行失敗: {e}")