import time
import os
import re
import pandas as pd
from FinMind.data import DataLoader
from datetime import datetime
from rapidfuzz import fuzz

# 設定參數
FINMIND_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMS0yNyAxNzo1MDoyOCIsInVzZXJfaWQiOiJBNzE0NDg4OCIsImVtYWlsIjoiQTcxNDQ4ODhAZ21haWwuY29tIiwiaXAiOiI0Mi43OS4xNjMuMjQzIn0._26kyEg1nWsvGAdbvPuzbOXcHuSWTR698SDSivTAy1M"
TARGET_STOCKS = ["2330"]
START_DATE = "2022-08-25"
END_DATE = "2023-12-31"
_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUTPUT_FILE = os.path.join(_DATA_RAW, f"old_news_{START_DATE}_to_{END_DATE}.csv")

# 初始化
dl = DataLoader(token=FINMIND_TOKEN)

def getNews(stock_id, start, end, filename):
    """抓取指定股票的新聞資料"""
    
    # 產生日期範圍（每天）
    date_list = pd.date_range(start=start, end=end, freq='D').strftime('%Y-%m-%d').tolist()
    
    print(f"\n開始抓取 {stock_id} 的新聞")
    print(f"時間範圍: {start} ~ {end}")
    print(f"總共需要查詢 {len(date_list)} 天\n")
    
    total_news = 0
    
    # 逐日查詢
    for i, date in enumerate(date_list, 1):
        try:
            # 呼叫API
            df = dl.get_data(
                dataset='TaiwanStockNews',
                data_id=stock_id,
                start_date=date,
                end_date=None
            )
            
            # 檢查是否有資料
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 轉換日期格式（保留時和分）
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
                
                # 確保 stock_id 為字串（保留前面ˇ的零）
                df['stock_id'] = df['stock_id'].astype(str)
                
                # 只保留查詢當天的資料
                df = df[df['date'].str.startswith(date)]
                
                # 刪除 link 欄位
                if 'link' in df.columns:
                    df = df.drop(columns=['link'])
                
                if not df.empty:
                    # 寫入CSV檔案
                    if os.path.isfile(filename):
                        df.to_csv(filename, mode='a', index=False, header=False, encoding='utf-8-sig')
                    else:
                        df.to_csv(filename, mode='w', index=False, header=True, encoding='utf-8-sig')
                    
                    print(f"[{i}/{len(date_list)}] {date}: {len(df)} 筆新聞")
                    total_news += len(df)
            
        except Exception as e:
            print(f"[{i}/{len(date_list)}] {date}: 錯誤 - {e}")
        
        # 避免請求過快
        time.sleep(0.3)
    
    print(f"\n{stock_id} 完成，共抓取 {total_news} 筆新聞\n")

def normalize_title(title):
    """清除標題中的空白、特殊符號、來源後綴，方便模糊比對"""
    if not isinstance(title, str):
        return ''
    # 移除「 - 來源名稱」結尾（如 - Yahoo奇摩新聞、- 經濟日報）
    title = re.sub(r'\s*-\s*\S+.*$', '', title)
    # 移除所有空白與標點符號（保留中英文與數字）
    title = re.sub(r'[\s\W]', '', title, flags=re.UNICODE)
    return title.strip()

def deduplicate_csv(filename, similarity_threshold=85):
    """
    去除CSV檔案中的重複新聞標題（模糊比對）
    similarity_threshold: 相似度門檻（0~100），超過此值視為重複，預設85
    """
    if not os.path.exists(filename):
        print(f"檔案不存在: {filename}")
        return
    
    print(f"\n正在處理 {filename} 的去重（相似度門檻: {similarity_threshold}）...")
    
    # 讀取CSV，指定 stock_id 為字串格式
    df = pd.read_csv(filename, dtype={'stock_id': str})
    original_count = len(df)
    
    # 按照 date 排序，確保保留最早的一筆
    df = df.sort_values(by=['stock_id', 'date']).reset_index(drop=True)
    
    # 擷取日期部分（yyyy-mm-dd）用於分組
    df['_date_only'] = df['date'].astype(str).str[:10]
    
    # 正規化標題用於比對
    df['_norm_title'] = df['title'].apply(normalize_title)
    
    keep_mask = [True] * len(df)
    
    # 只在同一天內進行模糊比對
    for _, group in df.groupby('_date_only'):
        indices = group.index.tolist()
        for ii, i in enumerate(indices):
            if not keep_mask[i]:
                continue
            for j in indices[ii + 1:]:
                if not keep_mask[j]:
                    continue
                score = fuzz.ratio(df.at[i, '_norm_title'], df.at[j, '_norm_title'])
                if score >= similarity_threshold:
                    keep_mask[j] = False
    
    df_deduped = df[keep_mask].drop(columns=['_date_only', '_norm_title']).reset_index(drop=True)
    
    # 寫回檔案
    df_deduped.to_csv(filename, index=False, encoding='utf-8-sig')
    
    removed_count = original_count - len(df_deduped)
    print(f"原始筆數: {original_count}")
    print(f"去重後: {len(df_deduped)} 筆")
    print(f"移除重複: {removed_count} 筆")
    print(f"檔案位置: {os.path.abspath(filename)}")




#主程式
for stock_id in TARGET_STOCKS:
    getNews(stock_id, START_DATE, END_DATE, OUTPUT_FILE)

# 對產生的CSV檔案進行去重
deduplicate_csv(OUTPUT_FILE)

print("\n完成!")