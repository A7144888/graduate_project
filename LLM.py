import pandas as pd
import ollama
import json
import os
import csv
import re

# --- 設定區 ---
INPUT_FILE = "old_news.csv" 
OUTPUT_FILE = "LLM_score_2024-1-1_2024-3-5.csv"
MODEL_NAME = "llama3.1:latest"
STOCK_ID = "2330"
SAVE_INTERVAL = 5 
MAX_RETRIES = 2  # 如果沒跑出分數，最多重試 2 次

def analyze_news(stock_id, title, content=None):
    """ 送交 Ollama 分析，加入重試機制與強化的 JSON 提取 """
    if content and str(content).strip() and str(content).lower() != 'nan':
        mode_desc = f"【新聞標題】：{title}\n    【新聞內文】：{content}"
    else:
        mode_desc = f"【新聞標題】：{title}\n    （無內文，請僅就標題進行推論）"

    prompt = f"""
    你是一位專業台股分析師。請分析以下新聞對股票代號 {stock_id} 的影響。
    {mode_desc}
    請嚴格以 JSON 格式回傳，不要有解釋文字。
    回傳格式：
    {{
        "sentiment_score": 介於 -1.0(極大利空) 與 1.0(極大利多) 之間的浮點數。
         "impact_intensity": 介於 0.0 與 1.0 之間的影響強度。
        "certainty": 介於 0.0 與 1.0 之間的確定性。
        "time_horizon": 【絕對禁止】回傳其他數值。必須且只能從 [0.8, 0.4, 0.1] 中選擇一個最接近的：
            - 0.8: 代表短期影響
            - 0.4: 代表中期影響
            - 0.1: 代表長期影響
        "summary": "一句話重點",
        "evidence": "具體事實或標題推論",
        "reason": "評分理由"
    }}
    """

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], format='json')
            res_content = response['message']['content'].strip()
            
            # 使用正則表達式提取 JSON 部分 (防止 LLM 在 JSON 前後講廢話)
            match = re.search(r'\{.*\}', res_content, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                return json.loads(res_content)
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"   ⚠️ 分析失敗，正在進行第 {attempt + 1} 次重試...")
                continue
            else:
                print(f"   ❌ 經過 {MAX_RETRIES} 次重試後仍然失敗: {str(e)[:50]}")
                return None

# --- 主程式執行 (其餘邏輯維持你的版本，僅修正儲存判斷) ---
try:
    if not os.path.exists(INPUT_FILE):
        print(f"💥 錯誤：找不到輸入檔案 {INPUT_FILE}")
    else:
        df_all = pd.read_csv(INPUT_FILE)
        total_all = len(df_all)
        
        start_row = 0
        if os.path.exists(OUTPUT_FILE):
            try:
                # 使用 pandas 讀取計數最準 (處理內文換行)
                df_count = pd.read_csv(OUTPUT_FILE, on_bad_lines='skip', engine='python')
                start_row = len(df_count)
                print(f"♻️  檢測到現有進度：已處理 {start_row} 筆，將從第 {start_row + 1} 筆開始。")
            except:
                start_row = 0

        df_todo = df_all.iloc[start_row:].copy()
        total_todo = len(df_todo)
        
        if total_todo == 0:
            print("🎉 所有新聞已處理完畢！")
        else:
            print(f"🚀 準備續寫，剩餘 {total_todo} 則新聞待處理...")
            temp_results = []
            current_count = 0

            for index, row in df_todo.iterrows():
                current_count += 1
                print(f"\n👉 [{current_count}/{total_todo}] 處理原始第 {index+1} 筆: {str(row['title'])[:25]}...")
                
                news_content = row.get('text') if 'text' in row else None
                analysis = analyze_news(STOCK_ID, row['title'], news_content)
                
                if analysis:
                    combined_data = {**row.to_dict(), **analysis}
                    temp_results.append(combined_data)
                    print(f"   ✅ 完成！分數: {analysis.get('sentiment_score')}")
                else:
                    # 如果連重試都失敗，填入預設值，避免聚合程式崩潰
                    error_fill = {
                        "sentiment_score": 0.0, "impact_intensity": 0.0, 
                        "certainty": 0.0, "time_horizon": 0.0,
                        "summary": "分析失敗", "evidence": "N/A", "reason": "LLM 解析錯誤"
                    }
                    temp_results.append({**row.to_dict(), **error_fill})
                    print(f"   🏮 已填入預設值 (分析失敗)")

                if current_count % SAVE_INTERVAL == 0 or current_count == total_todo:
                    try:
                        save_df = pd.DataFrame(temp_results)
                        file_exists = os.path.isfile(OUTPUT_FILE)
                        save_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
                        print(f"   💾 [進度儲存] 成功追加至第 {start_row + current_count} 筆。")
                        temp_results = [] 
                    except PermissionError:
                        print(f"   ❌ 寫入失敗！請關閉 Excel。")

except Exception as e:
    print(f"💥 程式執行失敗: {e}")