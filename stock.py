import yfinance as yf

data= yf.download(["2330.TW","0050.TW"], start='2020-01-01',auto_adjust=False)#沒寫end=則預設抓到今天

df_long= data.stack(level=1,future_stack=True).reset_index()#把level1的title移到垂直方向
df_long.rename(columns={'level_1':'Ticker'},inplace=True)#改欄位名改成Ticker

df_long= df_long.sort_values(by=['Ticker','Date'])
df_long.to_csv("stock.csv",index=False,encoding='utf-8-sig')
print(f"欄位: {df_long.columns.tolist()}")
print(f"資料型態: {df_long.dtypes}")
