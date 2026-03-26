import pandas as pd
import ollama
import json
import os
import csv

# --- 設定區 ---
INPUT_FILE = "old_news_2022-08-25_to_2023-12-31.csv" 
OUTPUT_FILE = "LLM_score_2022_2023.csv"
MODEL_NAME = "llama3.1:latest"
STOCK_ID = "2330"
SAVE_INTERVAL = 5 

def analyze_news(stock_id, title, content=None):
    """ 送交 Ollama 分析 """
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
        "sentiment_score": float,
        "impact_intensity": float,
        "certainty": float,
        "time_horizon": float,
        "summary": "一句話重點",
        "evidence": "具體事實或標題推論",
        "reason": "評分理由"
    }}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}], format='json')
        res_content = response['message']['content'].strip()
        if res_content.startswith("```json"):
            res_content = res_content[7:]
        if res_content.endswith("```"):
            res_content = res_content[:-3]
        return json.loads(res_content.strip())
    except Exception as e:
        print(f"   ❌ LLM 分析出錯: {e}")
        return None

# --- 主程式執行 ---
try:
    if not os.path.exists(INPUT_FILE):
        print(f"💥 錯誤：找不到輸入檔案 {INPUT_FILE}")
    else:
        # 1. 讀取原始資料
        df_all = pd.read_csv(INPUT_FILE)
        total_all = len(df_all)
        
        # 2. 斷點檢查：直接計算輸出檔已經有幾列
        start_row = 0
        if os.path.exists(OUTPUT_FILE):
            try:
                # 這裡不使用 pandas 讀取（避免格式錯誤卡死），改用純文字數行數
                with open(OUTPUT_FILE, 'r', encoding='utf-8-sig') as f:
                    row_count = sum(1 for line in f)
                
                # 扣除 header 欄位名那一行
                if row_count > 0:
                    start_row = row_count - 1
                
                print(f"♻️  檢測到現有進度：已處理 {start_row} 筆，將從第 {start_row + 1} 筆開始。")
            except Exception as e:
                print(f"⚠️ 讀取進度失敗，將從頭開始。錯誤：{e}")
                start_row = 0

        # 3. 擷取剩餘待跑的部分
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
                # 這裡的 index 就是原始 CSV 的行號
                print(f"\n👉 [{current_count}/{total_todo}] 處理原始第 {index+1} 筆: {str(row['title'])[:25]}...")
                
                news_content = row.get('text') if 'text' in row else None
                analysis = analyze_news(STOCK_ID, row['title'], news_content)
                
                if analysis:
                    combined_data = {**row.to_dict(), **analysis}
                    temp_results.append(combined_data)
                    print(f"   ✅ 完成！分數: {analysis.get('sentiment_score')}")
                else:
                    # 失敗也要補齊格式，確保行數對齊
                    temp_results.append(row.to_dict())

                # 4. 週期性存檔 (Append 模式)
                if current_count % SAVE_INTERVAL == 0 or current_count == total_todo:
                    try:
                        save_df = pd.DataFrame(temp_results)
                        file_exists = os.path.isfile(OUTPUT_FILE)
                        save_df.to_csv(
                            OUTPUT_FILE, 
                            mode='a', 
                            index=False, 
                            header=not file_exists, 
                            encoding='utf-8-sig',
                            quoting=csv.QUOTE_ALL
                        )
                        print(f"   💾 [進度儲存] 成功追加至第 {start_row + current_count} 筆。")
                        temp_results = [] 
                    except PermissionError:
                        print(f"   ❌ 寫入失敗！請關閉 Excel。資料保留在記憶體中...")

            print(f"\n✨ 任務完成！結果已續寫至 {OUTPUT_FILE}")

except Exception as e:
    print(f"💥 程式執行失敗: {e}")