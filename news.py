#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多來源財經新聞爬蟲（Selenium v2）
來源：Yahoo奇摩股市（限[Yahoo股市]來源）、經濟日報、天下雜誌、自由時報、中央社
"""

import re
import time
import logging
import requests as req
from datetime import datetime, timedelta
from urllib.parse import quote, urlparse

import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from newspaper import Article
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

logging.getLogger("jieba").setLevel(logging.ERROR)
logging.getLogger("newspaper").setLevel(logging.ERROR)

# ============================================================
# ★ 使用者設定
# ============================================================
KEYWORD    = "台積電"
START_DATE = "2026-02-20"   # YYYY-MM-DD（含）
END_DATE   = "2026-02-23"   # YYYY-MM-DD（含）
MAX_PAGES  = 5              # 各來源最多爬幾頁
DELAY      = 1.5            # 翻頁間隔（秒）
HEADLESS   = True           # False 可顯示瀏覽器視窗
MAX_RETRY  = 3              # 文章擷取最多重試次數
# ============================================================

start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}


# ──────────────────────────────────────────────────────────
# WebDriver 初始化
# ──────────────────────────────────────────────────────────
def make_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


# ──────────────────────────────────────────────────────────
# 日期工具
# ──────────────────────────────────────────────────────────
def in_range(date_str) -> bool:
    if not date_str:
        return False
    try:
        return start_dt <= datetime.strptime(str(date_str)[:10], "%Y-%m-%d") <= end_dt
    except Exception:
        return False


def parse_tw_date(text: str):
    """解析各種中文/英文日期格式，回傳 YYYY-MM-DD 或 None"""
    now = datetime.now()
    t = str(text).strip()
    m = re.search(r"(\d+)\s*(分鐘|小時)前", t)
    if m:
        n, u = int(m.group(1)), m.group(2)
        d = now - (timedelta(hours=n) if u == "小時" else timedelta(minutes=n))
        return d.strftime("%Y-%m-%d")
    if re.search(r"昨[天日]", t):
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    m = re.search(r"(\d+)\s*天前", t)
    if m:
        return (now - timedelta(days=int(m.group(1)))).strftime("%Y-%m-%d")
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", t)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", t)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    m = re.search(r"(\d{1,2})[月/](\d{1,2})[日]?", t)
    if m:
        return f"{now.year}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"
    return None


# ──────────────────────────────────────────────────────────
# 文章文字清理 & 關鍵字相關性
# ──────────────────────────────────────────────────────────

# 逐行雜訊 pattern（整行比對）
_NOISE_LINE_RE = re.compile(
    r"^("
    r"廣告[\s廣告]*"                        # 廣告 / 廣告 廣告
    r"|分享$"
    r"|Loading\.\.\."
    r"|展開全文.*"
    r"|免費看獨家內容.*"
    r"|登入會員.*"
    r"|免費試用\s*\d+\s*日.*"
    r"|立即啟動免費.*"
    r"|會員獨享.*"
    r"|全站內容隨你讀.*"
    r"|無廣告環境$"
    r"|產業資料庫$"
    r"|早安經濟日報$"
    r"|每日免費電子報.*"
    r"|收藏[、，]追蹤新聞.*"
    r"|免費註冊.*解鎖全文.*"
    r"|有限額度觀看.*"
    r"|剩\s*\d*\s*篇$"
    # ── 中央社（CNA）專屬雜訊 ──────────────────────────────
    r"|#\S+"                                # #hashtag 行
    r"|請同意我們的隱私權規範.*"            # 隱私聲明
    r"|（\d+/\d+\s+\d+:\d+\s*更新）"       # 更新時間，如（2/23 09:29 更新）
    r"|請繼續下滑閱讀"                      # 滾動提示
    r"|中央社.{0,6}一手新聞.{0,6}app"      # app 推廣
    r"|本網站之文字、圖片及影音.*"          # 版權聲明
    r"|新聞專題"                            # 麵包屑：頁面分類
    r"|首頁"                                # 麵包屑：首頁
    r"|/"                                   # 麵包屑：分隔符
    r"|\d{1,4}"                             # 麵包屑：頁數 / 筆數（純數字短行）
    r")$",
    re.I,
)

# 區塊雜訊 pattern（跨行移除）
_NOISE_BLOCK_PATTERNS = [
    re.compile(r"登入會員.{0,500}?解鎖全文", re.S),
    re.compile(r"展開全文.{0,800}?免費試用\s*\d+\s*日", re.S),
    re.compile(r"免費看獨家內容.{0,500}?立即啟動免費\s*\d+\s*日試閱", re.S),
    # 中央社：「延伸閱讀」及其後的推薦清單、相關文章列表、版權頁尾全部截除
    re.compile(r"延伸閱讀[\s\S]*", re.S),
]


def clean_text(text: str) -> str:
    """
    清理文章內文：
    1. 移除廣告、付費牆、訂閱提示等雜訊區塊
    2. 逐行移除雜訊行
    3. 收斂多餘空行
    """
    if not text:
        return text

    # 先移除跨行雜訊區塊
    for pat in _NOISE_BLOCK_PATTERNS:
        text = pat.sub("", text)

    # 逐行過濾
    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _NOISE_LINE_RE.match(stripped):
            continue
        clean_lines.append(line)

    text = "\n".join(clean_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # 最多連續兩個空行
    return text.strip()


def is_relevant(title: str, text: str, keyword: str) -> bool:
    """文章標題或內文含有關鍵字才視為相關"""
    return keyword in (title or "") or keyword in (text or "")


# ──────────────────────────────────────────────────────────
# 文章內文擷取
# ──────────────────────────────────────────────────────────
def _html_date(soup: BeautifulSoup, url: str):
    """
    備用日期提取（優先順序）：
    meta tag → <time datetime> → CNA URL 8碼日期 → URL路徑日期 → 頁面文字
    """
    # 1. meta tags
    for prop in ("article:published_time", "datePublished", "publishdate",
                 "pubdate", "og:updated_time", "DC.date.issued"):
        meta = soup.find("meta", property=prop) or soup.find("meta", {"name": prop})
        if meta and meta.get("content"):
            raw = meta["content"]
            d = raw.split("T")[0] if "T" in raw else raw[:10]
            if re.match(r"\d{4}-\d{2}-\d{2}", d):
                return d

    # 2. <time datetime="...">
    for t_el in soup.find_all("time", {"datetime": True}):
        raw = t_el["datetime"][:10]
        if re.match(r"\d{4}-\d{2}-\d{2}", raw):
            return raw

    # 3. CNA URL: /news/afe/202602230019.aspx
    m = re.search(r"/news/\w+/(20\d{2})(\d{2})(\d{2})\d+\.aspx", url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 4. 8碼日期嵌入 URL：YYYYMMDD 後接數字
    m = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?=\d)", url)
    if m:
        try:
            datetime.strptime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d")
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        except ValueError:
            pass

    # 5. URL 路徑中的 /YYYY/MM/DD/
    m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", url)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    # 6. 頁面文字中的 YYYY年MM月DD日
    m = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", soup.get_text())
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"

    return None


def _extract_with_requests(url: str, source: str):
    """requests + BeautifulSoup 備援擷取（newspaper3k 失敗時使用）"""
    try:
        resp = req.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # 標題
        title_el = soup.find("h1") or soup.find("h2") or soup.find("title")
        title = title_el.get_text(strip=True) if title_el else ""

        # 文章內文
        body = (
            soup.find("article")
            or soup.find("div", class_=re.compile(
                r"article.?(body|content|text)|news.?content|main.?content|"
                r"articleContent|article-content|story-body", re.I))
            or soup.find("div", id=re.compile(r"article|content|main", re.I))
        )
        if body:
            for tag in body.find_all(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = body.get_text(separator="\n", strip=True)
        else:
            text = ""

        if len(text) < 50:
            return None

        pub = _html_date(soup, url)
        return {"title": title, "text": text, "publish_date": pub,
                "source": source, "url": url}
    except Exception:
        return None


def extract_article(url: str, source: str = "") -> dict | None:
    """newspaper3k 擷取，失敗時改用 requests + BeautifulSoup；內文自動清理"""
    result = None
    for _ in range(MAX_RETRY):
        try:
            art = Article(url, language="zh")
            art.download()
            art.parse()
            if len(art.text) >= 50:
                pub = art.publish_date.strftime("%Y-%m-%d") if art.publish_date else None
                if pub is None:
                    pub = _html_date(BeautifulSoup(art.html, "html.parser"), url)
                result = {"title": art.title, "text": art.text, "publish_date": pub,
                          "source": source, "url": url}
                break
        except Exception:
            time.sleep(1)

    # newspaper3k 全部失敗 → requests 備援
    if result is None:
        result = _extract_with_requests(url, source)

    # 清理內文雜訊
    if result and result.get("text"):
        result["text"] = clean_text(result["text"])
        if len(result["text"]) < 50:   # 清理後內文過短 → 視為無效
            return None

    return result


# ──────────────────────────────────────────────────────────
# 1. Yahoo 奇摩股市（來源限 [Yahoo股市]）
# ──────────────────────────────────────────────────────────
def scrape_yahoo(driver: webdriver.Chrome, keyword: str) -> list[str]:
    """
    搜尋 Yahoo 奇摩財經新聞。
    優先用 Selenium XPath 定位「Yahoo股市」標籤再取同一區塊的連結；
    失敗時改用 BeautifulSoup 解析整頁。
    """
    print("[搜尋] Yahoo奇摩股市（來源限 Yahoo股市）...")
    links, seen = [], set()
    q = quote(keyword)

    for page in range(MAX_PAGES):
        b = page * 10 + 1
        url = (f"https://tw.news.search.yahoo.com/search"
               f"?p={q}&fr=finance&fr2=piv-web&b={b}")
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2.5)
        except Exception:
            break

        found_on_page = False

        # ── 方法一：Selenium XPath 直接定位 Yahoo股市 標籤 ──────────
        try:
            src_els = driver.find_elements(
                By.XPATH,
                "//*[normalize-space(text())='Yahoo股市' "
                "or contains(normalize-space(text()),'Yahoo股市')]"
            )
            for src_el in src_els:
                container = src_el
                for _ in range(8):
                    try:
                        container = driver.execute_script(
                            "return arguments[0].parentElement", container)
                        if container is None:
                            break
                        a_els = container.find_elements(
                            By.CSS_SELECTOR,
                            "a[href*='tw.stock.yahoo.com/news/']")
                        if a_els:
                            href = a_els[0].get_attribute("href")
                            p = urlparse(href)
                            clean = f"{p.scheme}://{p.netloc}{p.path}"
                            if clean not in seen:
                                seen.add(clean)
                                links.append(clean)
                                found_on_page = True
                            break
                    except Exception:
                        break
        except Exception:
            pass

        # ── 方法二：BeautifulSoup fallback ──────────────────────────
        if not found_on_page:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            # 嘗試各種容器型別
            containers = (
                soup.find_all("li", class_=re.compile(
                    r"js-stream-content|StreamMegaItem", re.I))
                or soup.find_all("div", class_=re.compile(
                    r"NewsArticle|stream|news-item", re.I))
                or soup.find_all("li")
            )
            for item in containers:
                item_text = item.get_text()
                if "Yahoo股市" not in item_text:
                    continue
                for a in item.find_all("a", href=True):
                    try:
                        p = urlparse(a["href"])
                        if (p.netloc == "tw.stock.yahoo.com"
                                and p.path.startswith("/news/")):
                            clean = f"{p.scheme}://{p.netloc}{p.path}"
                            if clean not in seen:
                                seen.add(clean)
                                links.append(clean)
                                found_on_page = True
                                break
                    except Exception:
                        pass

        if not found_on_page:
            break
        time.sleep(DELAY)

    print(f"  → {len(links)} 個連結")
    return links


# ──────────────────────────────────────────────────────────
# 2. 經濟日報
# ──────────────────────────────────────────────────────────
def scrape_udn(driver: webdriver.Chrome, keyword: str) -> list[str]:
    """經濟日報搜尋，URL 直接帶入日期區間"""
    print("[搜尋] 經濟日報...")
    links, seen = [], set()
    q = quote(keyword)

    for page in range(1, MAX_PAGES + 1):
        url = (f"https://money.udn.com/search/result/1001/{q}"
               f"?start_date={START_DATE}&end_date={END_DATE}&page={page}")
        try:
            driver.get(url)
            # 等待文章連結出現
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "a[href*='money.udn.com/money/story']")))
            except TimeoutException:
                time.sleep(3)
        except Exception:
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        pg_links = []
        for a in soup.find_all("a", href=True):
            h = a["href"]
            if "money.udn.com/money/story" in h and "/story/" in h:
                clean = h.split("?")[0]
                if clean not in seen:
                    seen.add(clean)
                    pg_links.append(clean)

        if not pg_links:
            break
        links.extend(pg_links)
        time.sleep(DELAY)

    print(f"  → {len(links)} 個連結")
    return links


# ──────────────────────────────────────────────────────────
# 3. 天下雜誌
# ──────────────────────────────────────────────────────────
def scrape_cw(driver: webdriver.Chrome, keyword: str) -> list[str]:
    """天下雜誌搜尋，依發布日在後置步驟篩選"""
    print("[搜尋] 天下雜誌...")
    links, seen = [], set()
    q = quote(keyword)

    for page in range(1, MAX_PAGES + 1):
        url = (f"https://www.cw.com.tw/search/doSearch.action"
               f"?key={q}&channel=all&sort=desc&page={page}")
        try:
            driver.get(url)
            # 等待文章連結出現
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "a[href*='/article/']")))
            except TimeoutException:
                time.sleep(3)
        except Exception:
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        pg_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"] or ""
            if href.startswith("/"):
                href = "https://www.cw.com.tw" + href
            try:
                p = urlparse(href)
                if p.netloc == "www.cw.com.tw" and "/article/" in p.path:
                    clean = f"{p.scheme}://{p.netloc}{p.path}"
                    if clean not in seen:
                        seen.add(clean)
                        pg_links.append(clean)
            except Exception:
                pass

        if not pg_links:
            break
        links.extend(pg_links)
        time.sleep(DELAY)

    print(f"  → {len(links)} 個連結")
    return links


# ──────────────────────────────────────────────────────────
# 4. 自由時報（SPA，需等 JS 渲染）
# ──────────────────────────────────────────────────────────
def scrape_ltn(driver: webdriver.Chrome, keyword: str) -> list[str]:
    """
    自由時報搜尋。search.ltn.com.tw 是 SPA。
    正確路徑為 /list，類型為 business（財經）。
    """
    print("[搜尋] 自由時報...")
    links, seen = [], set()
    q = quote(keyword)
    s = START_DATE.replace("-", "")
    e = END_DATE.replace("-", "")

    for page in range(1, MAX_PAGES + 1):
        url = (f"https://search.ltn.com.tw/list"
               f"?keyword={q}&type=business&sort=date"
               f"&start_time={s}&end_time={e}&page={page}")
        try:
            driver.get(url)
            # SPA 需等 JS 渲染：等待任何 ltn.com.tw 文章連結出現
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR,
                         "a[href*='news.ltn.com.tw'], a[href*='ec.ltn.com.tw']")))
            except TimeoutException:
                # 若等不到，多等一段讓 SPA 渲染
                time.sleep(4)
        except Exception:
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        pg_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"] or ""
            try:
                p = urlparse(href)
                if ("ltn.com.tw" in p.netloc
                        and p.netloc != "search.ltn.com.tw"):
                    # 文章 URL 路徑結尾必須是數字 ID（排除分類列表頁）
                    if re.search(r"/\d{6,}$", p.path):
                        clean = f"{p.scheme}://{p.netloc}{p.path}"
                        if clean not in seen:
                            seen.add(clean)
                            pg_links.append(clean)
            except Exception:
                pass

        if not pg_links:
            break
        links.extend(pg_links)
        time.sleep(DELAY)

    print(f"  → {len(links)} 個連結")
    return links


# ──────────────────────────────────────────────────────────
# 5. 中央社
# ──────────────────────────────────────────────────────────
# CNA URL 格式：/news/CATEGORY/YYYYMMDDNNNN.aspx
_CNA_DATE_RE = re.compile(
    r"/news/\w+/(20\d{2})(\d{2})(\d{2})\d+\.aspx", re.I)
# 搜尋結果連結文字尾綴為「YYYY/MM/DD HH:MM」，導覽列連結文字無此尾綴
_CNA_DATE_SUFFIX_RE = re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}$")


def _cna_url_date(href: str):
    """從 CNA URL 路徑提取 YYYY-MM-DD；非新聞文章 URL 回傳 None"""
    m = _CNA_DATE_RE.search(href)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def scrape_cna(driver: webdriver.Chrome, keyword: str) -> list[str]:
    """
    中央社搜尋。
    使用 BS-date-suffix 方法：搜尋結果連結文字以「YYYY/MM/DD HH:MM」結尾，
    導覽列連結文字無此尾綴，藉此精準區分。
    """
    print("[搜尋] 中央社...")
    links, seen = [], set()
    q = quote(keyword)

    for page in range(0, MAX_PAGES):
        url = f"https://www.cna.com.tw/search/hysearchws.aspx?q={q}&page={page}"
        try:
            driver.get(url)
            try:
                WebDriverWait(driver, 12).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR,
                         "a[href*='/news/'][href$='.aspx']")))
            except TimeoutException:
                time.sleep(3)
            time.sleep(1.5)
        except Exception:
            break

        pg_links = []
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"] or ""
            if href.startswith("/"):
                href = "https://www.cna.com.tw" + href
            if not _CNA_DATE_SUFFIX_RE.search(a.get_text(strip=True)):
                continue
            url_date = _cna_url_date(href)
            if url_date is None or not in_range(url_date):
                continue
            try:
                p = urlparse(href)
                if (p.netloc == "www.cna.com.tw"
                        and "/news/" in p.path
                        and p.path.endswith(".aspx")):
                    clean = f"{p.scheme}://{p.netloc}{p.path}"
                    if clean not in seen:
                        seen.add(clean)
                        pg_links.append(clean)
            except Exception:
                pass

        print(f"  [CNA] page={page}: {len(pg_links)} 個")

        if not pg_links:
            break
        links.extend(pg_links)
        time.sleep(DELAY)

    print(f"  → {len(links)} 個連結")
    return links


# ──────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"關鍵字：{KEYWORD}　區間：{START_DATE} ～ {END_DATE}\n")

    # ── 收集各來源 URL ────────────────────────────────────
    driver = make_driver()
    # (url, 來源名稱, 來自日期過濾來源?)
    source_triples: list[tuple[str, str, bool]] = []
    try:
        for u in scrape_yahoo(driver, KEYWORD):
            source_triples.append((u, "Yahoo股市", False))
        for u in scrape_udn(driver, KEYWORD):
            source_triples.append((u, "經濟日報", True))   # URL 已含日期過濾
        for u in scrape_cw(driver, KEYWORD):
            source_triples.append((u, "天下雜誌", False))
        for u in scrape_ltn(driver, KEYWORD):
            source_triples.append((u, "自由時報", True))   # URL 已含日期過濾
        for u in scrape_cna(driver, KEYWORD):
            source_triples.append((u, "中央社", False))
    finally:
        driver.quit()

    # ── URL 去重 ──────────────────────────────────────────
    seen_urls: set[str] = set()
    unique: list[tuple[str, str, bool]] = []
    for u, src, date_filtered in source_triples:
        if u not in seen_urls:
            seen_urls.add(u)
            unique.append((u, src, date_filtered))

    print(f"\n共找到 {len(unique)} 篇文章連結，開始擷取內文...\n")

    # ── 擷取內文 ──────────────────────────────────────────
    data = []
    skipped_irrelevant = 0
    for u, src, date_filtered in tqdm(unique):
        art = extract_article(u, src)
        if art:
            # 若 publish_date 為空且來自有日期過濾的來源，以 START_DATE 填補
            if not art["publish_date"] and date_filtered:
                art["publish_date"] = START_DATE
            # 過濾與關鍵字無關的文章
            if not is_relevant(art.get("title", ""), art.get("text", ""), KEYWORD):
                skipped_irrelevant += 1
                continue
            data.append(art)
        time.sleep(DELAY)

    if skipped_irrelevant:
        print(f"[過濾] 排除與關鍵字無關的文章 {skipped_irrelevant} 篇")

    # ── 整理 DataFrame ────────────────────────────────────
    cols = ["title", "text", "publish_date", "source", "url"]
    df = pd.DataFrame(data, columns=cols) if data else pd.DataFrame(columns=cols)

    if df.empty:
        print("\n[結果] 此關鍵字與日期區間內沒有成功解析的新聞。")
    else:
        df.drop_duplicates(subset=["title"], inplace=True)

        # 依發布日篩選：有日期者需在區間內；無日期者保留（來源已做過濾）
        df["_d"] = df["publish_date"].apply(
            lambda s: str(s)[:10] if pd.notna(s) and str(s).strip() else "")
        before = len(df)
        df = df[
            ((df["_d"] >= START_DATE) & (df["_d"] <= END_DATE))
            | (df["_d"] == "")         # 無日期但已通過來源篩選者保留
        ].copy()
        df.drop(columns=["_d"], inplace=True)
        print(f"[篩選] 依發布日保留 {len(df)} 則"
              f"（排除 {before - len(df)} 則區間外）")

        # 預覽前 3 則
        print("\n========== 前 3 則新聞 ==========")
        for i in range(min(3, len(df))):
            r = df.iloc[i]
            print(f"\n--- 第 {i+1} 則 ---")
            print("來源：", r["source"])
            print("標題：", r["title"])
            print("日期：", r["publish_date"])
            print("網址：", r["url"])
            print("內文前 200 字：")
            print(str(r["text"])[:200])
            print("================================")

    # ── 存檔 ──────────────────────────────────────────────
    filename = f"news_{KEYWORD}_{START_DATE}_to_{END_DATE}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n成功寫入 CSV 共 {len(df)} 則，檔名：{filename}")
    print(f"欄位: {df.columns.tolist()}")
    print(f"資料型態:\n {df.dtypes}")
