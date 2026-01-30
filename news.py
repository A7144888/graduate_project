import time
import os
import pandas as pd
from FinMind.data import DataLoader
from datetime import datetime

# 設定參數
FINMIND_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMS0yNyAxNzo1MDoyOCIsInVzZXJfaWQiOiJBNzE0NDg4OCIsImVtYWlsIjoiQTcxNDQ4ODhAZ21haWwuY29tIiwiaXAiOiI0Mi43OS4xNjMuMjQzIn0._26kyEg1nWsvGAdbvPuzbOXcHuSWTR698SDSivTAy1M"
TARGET_STOCKS = ["0050"]
START_DATE = "2024-12-01"
END_DATE = "2024-12-02"
OUTPUT_FILE = "news.csv"

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
                # 轉換日期格式
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                # 確保 stock_id 為字串（保留前面ˇ的零）
                df['stock_id'] = df['stock_id'].astype(str)
                
                # 只保留查詢當天的資料
                df = df[df['date'] == date]
                
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




#主程式
for stock_id in TARGET_STOCKS:
    getNews(stock_id, START_DATE, END_DATE, OUTPUT_FILE)

# 去重 排序
if os.path.exists(OUTPUT_FILE):
    print("正在處理資料...")
    
    # 讀指定 stock_id 為字串格式，保留前面的零
    df = pd.read_csv(OUTPUT_FILE, dtype={'stock_id': str})
    original_count = len(df)
    
    df = df.drop_duplicates(subset=['stock_id', 'date', 'title'])
    df = df.sort_values(by=['stock_id', 'date'])
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 顯示結果
    print(f"原始筆數: {original_count}")
    print(f"去重後: {len(df)} 筆")
    print(f"檔案位置: {os.path.abspath(OUTPUT_FILE)}")
    print("\n完成!")
    print(f"欄位: {df.columns.tolist()}")
    print(f"資料型態: {df.dtypes}")

else:
    print("未產生CSV檔案")
