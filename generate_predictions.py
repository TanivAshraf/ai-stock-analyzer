# --- generate_predictions.py (EXPANDED 8-ASSET QUANTITATIVE VERSION) ---

import os
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import time
import csv

# --- Configuration: Cross-Sector Universe ---
SYMBOLS = ['AAPL', 'NVDA', 'MSFT', 'JPM', 'JNJ', 'XOM', 'SPY', 'TSLA']

SECTOR_MAP = {
    'AAPL': 'Tech Hardware',
    'NVDA': 'Semiconductors',
    'MSFT': 'Enterprise Cloud',
    'JPM': 'Financials',
    'JNJ': 'Healthcare',
    'XOM': 'Energy',
    'SPY': 'Market Index',
    'TSLA': 'Consumer Disc.'
}

LIVE_JSON_FILE = 'predictions.json'
HISTORY_CSV_FILE = 'history.csv'

# --- API Credentials ---
raw_gemini_key = os.environ.get('GEMINI_API_KEY', '')
GEMINI_API_KEY = raw_gemini_key.strip().strip('"').strip("'")

raw_news_key = os.environ.get('NEWS_API_KEY', '')
NEWS_API_KEY = raw_news_key.strip().strip('"').strip("'")

if not GEMINI_API_KEY:
    print("FATAL: GEMINI_API_KEY not found in environment secrets.")
    exit(1)

# --- Dynamic Model Resolution ---
def resolve_active_gemini_endpoint():
    """Verifies live model availability, prioritizing gemini-3.6-flash."""
    candidates = ['gemini-3.6-flash', 'gemini-2.5-pro']
    
    for candidate in candidates:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent"
        try:
            ping_url = f"{endpoint}?key={GEMINI_API_KEY}"
            test_res = requests.post(
                ping_url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": "ping"}]}]},
                timeout=10
            )
            if test_res.status_code == 200:
                print(f"Verified live operational model: {candidate}")
                return endpoint
            else:
                print(f"Model {candidate} ping returned status {test_res.status_code}")
        except Exception as e:
            print(f"Model {candidate} ping check error: {e}")

    print("Defaulting to recommended: gemini-3.6-flash")
    return "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

ACTIVE_API_URL = resolve_active_gemini_endpoint()

# --- Technical Indicators Calculation ---
def compute_technical_indicators(close_series):
    """Calculates 14-day RSI, 20-day SMA, 50-day SMA, and 20-day Annualized Volatility."""
    try:
        sma_20 = close_series.rolling(window=min(20, len(close_series))).mean().iloc[-1]
        sma_50 = close_series.rolling(window=min(50, len(close_series))).mean().iloc[-1]
        
        # 14-Day RSI
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 20-Day Annualized Volatility
        returns = close_series.pct_change()
        vol_20 = returns.tail(20).std() * (252 ** 0.5) * 100
        
        return {
            "sma_20": round(float(sma_20), 2),
            "sma_50": round(float(sma_50), 2),
            "rsi_14": round(float(rsi_14), 1),
            "volatility_ann_pct": round(float(vol_20), 1)
        }
    except Exception as e:
        print(f"Technical indicator error: {e}")
        return {"sma_20": "N/A", "sma_50": "N/A", "rsi_14": "N/A", "volatility_ann_pct": "N/A"}

# --- Data Ingestion ---
def get_stock_data_and_news(symbol):
    """Fetches historical stock data and recent news headlines."""
    stock_data = yf.download(symbol, period="3mo", auto_adjust=True, progress=False)
    if stock_data.empty or len(stock_data) < 2:
        raise ValueError(f"yfinance returned insufficient data for {symbol}")
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        try:
            close_series = stock_data['Close'][symbol]
        except KeyError:
            close_series = stock_data['Close'].iloc[:, 0]
    else:
        close_series = stock_data['Close']

    news_headlines = "No recent news found."
    if NEWS_API_KEY:
        try:
            search_term = symbol if symbol != 'SPY' else 'S&P 500'
            news_url = (
                f"https://newsapi.org/v2/everything?q={search_term}&language=en"
                f"&sortBy=publishedAt&pageSize=8&apiKey={NEWS_API_KEY}"
            )
            res = requests.get(news_url, timeout=12)
            res.raise_for_status()
            articles = res.json().get('articles', [])
            if articles:
                news_headlines = "\n".join([f"- {a['title']}" for a in articles if a.get('title')])
        except Exception:
            news_headlines = "Could not fetch news headlines."

    return stock_data, close_series, news_headlines

# --- Quantitative LLM Analysis ---
def get_ai_analysis(symbol, historical_data, close_series, news_headlines):
    """Prompts Gemini with price action, technical indicators, and news."""
    tech = compute_technical_indicators(close_series)
    sector = SECTOR_MAP.get(symbol, 'Market Asset')

    prompt = f"""
    You are an expert quantitative hedge-fund analyst. Analyze {symbol} ({sector}).
    
    Calculated Technical Indicators:
    - 14-Day RSI: {tech['rsi_14']} (Overbought > 70, Oversold < 30)
    - 20-Day SMA: ${tech['sma_20']} vs 50-Day SMA: ${tech['sma_50']}
    - 20-Day Annualized Volatility: {tech['volatility_ann_pct']}%

    Recent 30-Day Close & Volume Context:
    {historical_data[['Close', 'Volume']].tail(25).to_string()}

    Recent News Headlines:
    {news_headlines}

    Respond with a strictly valid single JSON object containing exactly these keys:
    - "sentiment": string, strictly one of "Bullish", "Bearish", or "Neutral".
    - "confidence": float between 0.50 and 1.00 indicating signal conviction.
    - "primary_driver": strictly one of ["Earnings/Financials", "Macro/Fed", "Product/Technology", "Regulatory", "Technical Momentum"].
    - "predicted_range": array of two numbers [predicted_low, predicted_high] for tomorrow's trading session.
    - "reasoning": string, concise 2-sentence rationale synthesizing price action, technical indicators, and news.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    target_url = f"{ACTIVE_API_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(target_url, headers=headers, json=payload, timeout=45)
            
            if response.status_code != 200:
                safe_err = response.text.replace(GEMINI_API_KEY, "[REDACTED]")
                print(f"API Error Response ({response.status_code}): {safe_err}")
                response.raise_for_status()
            
            res_json = response.json()
            candidates = res_json.get('candidates', [])
            if not candidates:
                raise ValueError("No candidate generation returned by model.")
            
            raw_text = candidates[0]['content']['parts'][0]['text'].strip()
            clean_text = raw_text.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(clean_text)
            
            # Normalize fields
            sentiment = str(parsed.get('sentiment', 'Neutral')).capitalize()
            if sentiment not in ['Bullish', 'Bearish', 'Neutral']:
                sentiment = 'Neutral'
            parsed['sentiment'] = sentiment

            try:
                conf = float(parsed.get('confidence', 0.65))
                parsed['confidence'] = max(0.50, min(1.00, round(conf, 2)))
            except (ValueError, TypeError):
                parsed['confidence'] = 0.65

            valid_drivers = ["Earnings/Financials", "Macro/Fed", "Product/Technology", "Regulatory", "Technical Momentum"]
            driver = parsed.get('primary_driver', 'Technical Momentum')
            parsed['primary_driver'] = driver if driver in valid_drivers else "Technical Momentum"

            return parsed

        except Exception as e:
            safe_exception = str(e).replace(GEMINI_API_KEY, "[REDACTED]")
            print(f"Attempt {attempt + 1} for {symbol} failed: {safe_exception}")
            if attempt < max_retries - 1:
                time.sleep(4)
            else:
                raise RuntimeError(f"All retry attempts failed for {symbol}.")

# --- Logging & Persistent Storage ---
def log_to_history_csv(log_data):
    """Appends evaluation row to history.csv, maintaining backward compatibility."""
    headers = [
        'date', 'symbol', 'actual_price', 'price_change', 
        'price_change_percent', 'ai_sentiment_for_tomorrow',
        'confidence', 'primary_driver',
        'predicted_low_for_tomorrow', 'predicted_high_for_tomorrow',
        'yesterdays_predicted_range', 'accuracy_check_hit'
    ]
    
    file_exists = os.path.isfile(HISTORY_CSV_FILE)
    
    # Auto-upgrade header if existing file has legacy 10-column header
    if file_exists:
        try:
            with open(HISTORY_CSV_FILE, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().split(',')
            if 'confidence' not in first_line:
                # Read all lines and rewrite with upgraded header
                with open(HISTORY_CSV_FILE, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                all_lines[0] = ','.join(headers) + '\n'
                with open(HISTORY_CSV_FILE, 'w', encoding='utf-8') as f:
                    f.writelines(all_lines)
        except Exception as e:
            print(f"CSV header upgrade notice: {e}")

    with open(HISTORY_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)

# --- Main Pipeline ---
def main():
    previous_predictions = {}
    try:
        with open(LIVE_JSON_FILE, 'r', encoding='utf-8') as f:
            prev_data = json.load(f)
            for item in prev_data.get('predictions', []):
                if 'symbol' in item and 'error' not in item:
                    previous_predictions[item['symbol']] = item
    except (FileNotFoundError, json.JSONDecodeError):
        print("Previous predictions file not found or empty. Initializing new run.")

    todays_data_for_json = {
        'last_updated': datetime.now(timezone.utc).isoformat(),
        'predictions': []
    }

    for symbol in SYMBOLS:
        print(f"Processing {symbol} ({SECTOR_MAP.get(symbol)})...")
        try:
            if SYMBOLS.index(symbol) > 0:
                time.sleep(4)  # Respect free tier rate limits

            stock_data, close_series, news = get_stock_data_and_news(symbol)
            current_price = float(close_series.iloc[-1])
            previous_close = float(close_series.iloc[-2])
            
            ai_output = get_ai_analysis(symbol, stock_data, close_series, news)
            
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close) * 100
            
            # Historical accuracy check from yesterday
            accuracy_check_hit = None
            yesterdays_predicted_range_str = "N/A"
            if symbol in previous_predictions:
                yest_pred = previous_predictions[symbol]
                p_range = yest_pred.get('predicted_range')
                if p_range and len(p_range) == 2 and p_range[0] is not None and p_range[1] is not None:
                    low, high = float(p_range[0]), float(p_range[1])
                    accuracy_check_hit = (low <= current_price <= high)
                    yesterdays_predicted_range_str = f"${low:.2f} - ${high:.2f}"
            
            pred_range = ai_output.get('predicted_range') or [None, None]
            low_val = round(float(pred_range[0]), 2) if pred_range[0] is not None else None
            high_val = round(float(pred_range[1]), 2) if pred_range[1] is not None else None

            # 1. Update Live JSON for Frontend
            live_record = {
                'symbol': symbol,
                'sector': SECTOR_MAP.get(symbol, 'General'),
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_percent': round(price_change_percent, 2),
                'sentiment': ai_output.get('sentiment'),
                'confidence': ai_output.get('confidence', 0.65),
                'primary_driver': ai_output.get('primary_driver', 'Technical Momentum'),
                'reasoning': ai_output.get('reasoning'),
                'predicted_range': [low_val, high_val],
                'accuracy_check': {
                    "yesterdays_predicted_range": yesterdays_predicted_range_str,
                    "todays_actual_price": f"${current_price:.2f}",
                    "hit": accuracy_check_hit
                } if accuracy_check_hit is not None else None
            }
            todays_data_for_json['predictions'].append(live_record)

            # 2. Append to Persistent History CSV for Quantitative Backtests
            historical_log_record = {
                'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'symbol': symbol,
                'actual_price': round(current_price, 4),
                'price_change': round(price_change, 4),
                'price_change_percent': round(price_change_percent, 4),
                'ai_sentiment_for_tomorrow': ai_output.get('sentiment'),
                'confidence': ai_output.get('confidence', 0.65),
                'primary_driver': ai_output.get('primary_driver', 'Technical Momentum'),
                'predicted_low_for_tomorrow': low_val,
                'predicted_high_for_tomorrow': high_val,
                'yesterdays_predicted_range': yesterdays_predicted_range_str,
                'accuracy_check_hit': accuracy_check_hit
            }
            log_to_history_csv(historical_log_record)

            print(f"Successfully processed {symbol}: {ai_output.get('sentiment')} | Conf: {ai_output.get('confidence')} | Driver: {ai_output.get('primary_driver')}")

        except Exception as err:
            print(f"Error processing {symbol}: {err}")
            todays_data_for_json['predictions'].append({
                'symbol': symbol,
                'sector': SECTOR_MAP.get(symbol, 'General'),
                'error': 'Data or prediction pipeline failure for this session.'
            })

    with open(LIVE_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(todays_data_for_json, f, indent=4)
        
    print(f"\nCompleted execution across {len(SYMBOLS)} assets. Updated {LIVE_JSON_FILE} and {HISTORY_CSV_FILE}.")

if __name__ == "__main__":
    main()
