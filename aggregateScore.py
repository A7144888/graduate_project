import pandas as pd
import numpy as np

# --- 設定區 ---
INPUT_FILE = "LLM_score.csv"
OUTPUT_FILE = "daily_features.csv"

def aggregate_daily_news(group):
    """
    針對每一天的所有新聞進行特徵聚合計算
    """
    # 1. 基礎準備：計算單則新聞的原始得分 (sentiment * intensity * certainty)
    # 使用 .abs() 確保排序是基於「影響力絕對大小」
    group = group.copy()
    group['raw_score'] = group['sentiment_score'] * group['impact_intensity'] * group['certainty']
    group['abs_intensity'] = group['impact_intensity'].abs()

    # 2. 計算 Net Impact (階梯式衰減加總)
    # 按 impact_intensity 從大到小排序
    sorted_news = group.sort_values(by='impact_intensity', ascending=False)
    scores = sorted_news['raw_score'].values
    
    # 建立衰減權重: 1.0, 0.5, 0.25, 0.125...
    weights = 0.5 ** np.arange(len(scores))
    net_impact = np.sum(scores * weights)

    # 3. 計算 Weighted Horizon (以 impact_intensity 為權重的加權平均)
    total_intensity = group['impact_intensity'].sum()
    if total_intensity > 0:
        weighted_horizon = (group['time_horizon'] * group['impact_intensity']).sum() / total_intensity
    else:
        weighted_horizon = 0  # 防呆：若當天強度總和為 0

    # 4. 計算 Divergence (Sentiment 的標準差)
    # 如果只有一則新聞，標準差定義為 0
    divergence = group['sentiment_score'].std() if len(group) > 1 else 0.0

    return pd.Series({
        'net_impact': net_impact,
        'weighted_horizon': weighted_horizon,
        'news_count': int(len(group)),
        'divergence': divergence
    })

def main():
    try:
        # 讀取 LLM 分析後的 CSV
        df = pd.read_csv(INPUT_FILE)
        
        # 轉換日期格式（僅保留日期部分，去除小時分鐘，以便按天分組）
        # 假設 publish_date 格式為 "2026-02-23 09:18"
        df['date_only'] = pd.to_datetime(df['publish_date']).dt.date
        
        print(f"📊 正在聚合數據...")
        
        # 按日期分組並套用聚合函數
        daily_df = df.groupby('date_only').apply(aggregate_daily_news).reset_index()
        
        # 重新命名日期欄位
        daily_df.rename(columns={'date_only': 'date'}, inplace=True)
        
        # 儲存結果
        daily_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"✨ 聚合完成！")
        print(f"📅 處理日期區間: {daily_df['date'].min()} ~ {daily_df['date'].max()}")
        print(f"💾 每日特徵已存至: {OUTPUT_FILE}")
        print("\n預覽前幾筆數據：")
        print(daily_df.head())

    except Exception as e:
        print(f"💥 程式執行失敗: {e}")

if __name__ == "__main__":
    main()