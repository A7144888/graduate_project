import pandas as pd
import ollama
import json
import cloudscraper
import requests
from bs4 import BeautifulSoup

# --- 設定區 ---
INPUT_FILE = "news.csv"
OUTPUT_FILE = "LLM_score.csv"
MODEL_NAME = "llama3.1:latest"
# TEST_COUNT 已移除，改為全量處理

# 初始化爬蟲器
scraper = cloudscraper.create_scraper()

def get_real_url(google_url):
    """ 直接追蹤 Google News 的跳轉，獲取真實網址 """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(google_url, headers=headers, timeout=10, allow_redirects=True)
        real_url = response.url
        
        if "news.google.com" not in real_url:
            return real_url
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            target_link = soup.find('a', {'jslog': True})
            if target_link and target_link.get('href'):
                return target_link['href']
    except Exception as e:
        print(f"   ⚠️ 跳轉追蹤失敗: {e}")
    return google_url

def fetch_news_content_stable(url):
    """ 獲取真實網址後抓取內文 """
    try:
        real_url = get_real_url(url)
        if "news.google.com" in real_url and len(real_url) < 200:
            return ""
            
        print(f"   🔗 成功跳轉至: {real_url[:60]}...")
        
        response = scraper.get(real_url, timeout=12)
        response.encoding = response.apparent_encoding
        
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
            s.extract()

        content_selectors = ['article', '.article-content', '.caas-body', '.story', '.news-content']
        text = ""
        for sel in content_selectors:
            target = soup.select_one(sel)
            if target:
                text = target.get_text(separator=' ', strip=True)
                break
        
        if not text:
            text = " ".join([p.get_text(strip=True) for p in soup.find_all(['p', 'div']) if len(p.get_text(strip=True)) > 25])

        clean_text = " ".join(text.split())
        return clean_text[:2000] if len(clean_text) > 80 else ""

    except Exception as e:
        print(f"   ⚠️ 抓取異常: {e}")
        return ""

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
    df_all = pd.read_csv(INPUT_FILE)
    total_count = len(df_all)
    print(f"🚀 開始全量處理，共 {total_count} 則新聞 ...")

    results = []
    for index, row in df_all.iterrows():
        print(f"\n👉 [{index+1}/{total_count}] 處理中: {row['stock_id']} - {row['title'][:15]}...")
        
        full_text = fetch_news_content_stable(row['link'])
        
        if full_text:
            print(f"   🔍 成功抓取內文：{len(full_text)} 字")
        else:
            print(f"   ⚠️ 抓取失敗 (字數為 0)")

        analysis = analyze_full_news(row['stock_id'], row['title'], full_text)
        
        if analysis:
            results.append({**row.to_dict(), **analysis})
            print(f"   ✅ 分析完成！情緒: {analysis.get('sentiment_score')} | 時效: {analysis.get('time_horizon')}")
        else:
            results.append(row.to_dict())

    # 儲存結果
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✨ 任務完成！結果已存至 {OUTPUT_FILE}")

except Exception as e:
    print(f"💥 程式執行失敗: {e}")