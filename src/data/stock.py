import os
import yfinance as yf

_DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

stocks=['2330.TW']
start_date='2026-03-04'
end_date='2026-03-07'
for stock in stocks:
    data= yf.download(stock, start=start_date,end=end_date,auto_adjust=False)#沒寫end=則預設抓到今天

    df_long= data.stack(level=1,future_stack=True).reset_index()#把level1的title移到垂直方向
    df_long.rename(columns={'level_1':'Ticker'},inplace=True)#改欄位名改成Ticker

    df_long= df_long.sort_values(by=['Ticker','Date'])
    out_path = os.path.join(_DATA_RAW, f"stock_{stock}_{start_date}_to_{end_date}.csv")
    df_long.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"stock_{stock}_{start_date}_to_{end_date}.csv 已保存")
    print(f"欄位: {df_long.columns.tolist()}")
    print(f"資料型態:\n {df_long.dtypes}")
