#!/usr/bin/env python3
"""
Working data collection script with your functional APIs.
"""

import sys
import os
sys.path.append('.')

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
import time

# Your working API keys
ALPHA_VANTAGE_KEY = "H8U3J9QXHBRSM1KU"
NEWSAPI_KEY = "799d9f41a7c240d18c06860ca1ffa31e"

def collect_yahoo_data(symbols, period="1y"):
    """Collect data from Yahoo Finance."""
    print("📊 Collecting Yahoo Finance data...")
    data_dict = {}
    
    for symbol in symbols:
        try:
            print(f"  📈 Fetching {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                # Clean data
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                data.reset_index(inplace=True)
                data['symbol'] = symbol
                
                data_dict[symbol] = data
                print(f"    ✅ Success: {len(data)} records")
                
                # Save individual file
                os.makedirs("data/raw", exist_ok=True)
                filename = f"data/raw/{symbol}_yahoo_1y.csv"
                data.to_csv(filename, index=False)
                print(f"    💾 Saved: {filename}")
            else:
                print(f"    ❌ No data for {symbol}")
                
        except Exception as e:
            print(f"    ❌ Error for {symbol}: {e}")
        
        time.sleep(0.5)  # Be nice to the API
    
    return data_dict

def collect_alpha_vantage_data(symbols, limit=3):
    """Collect data from Alpha Vantage (limited calls)."""
    print("📊 Collecting Alpha Vantage data...")
    print(f"⚠️  Limited to {limit} symbols due to free tier (25 calls/day)")
    
    data_dict = {}
    base_url = "https://www.alphavantage.co/query"
    
    for i, symbol in enumerate(symbols[:limit]):
        try:
            print(f"  📈 Fetching {symbol} ({i+1}/{limit})...")
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': ALPHA_VANTAGE_KEY,
                'outputsize': 'compact'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame(data['Time Series (Daily)']).T
                df.columns = [col.split('. ')[1].lower().replace(' ', '_') for col in df.columns]
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'date'}, inplace=True)
                df['symbol'] = symbol
                
                # Convert to numeric
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])
                
                data_dict[symbol] = df
                print(f"    ✅ Success: {len(df)} records")
                
                # Save individual file
                os.makedirs("data/raw", exist_ok=True)
                filename = f"data/raw/{symbol}_alphavantage_daily.csv"
                df.to_csv(filename, index=False)
                print(f"    💾 Saved: {filename}")
                
            elif 'Note' in data:
                print(f"    ⚠️  Rate limit hit: {data['Note']}")
                break
            else:
                print(f"    ❌ Error: {data}")
            
        except Exception as e:
            print(f"    ❌ Error for {symbol}: {e}")
        
        # Rate limiting - Alpha Vantage free tier: 5 calls per minute
        if i < limit - 1:
            print("    ⏳ Waiting 12 seconds (rate limit)...")
            time.sleep(12)
    
    return data_dict

def collect_news_data(query="stock market", days=7):
    """Collect news data from NewsAPI."""
    print("📰 Collecting news data...")
    
    try:
        base_url = "https://newsapi.org/v2/everything"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'q': query,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'pageSize': 50
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        data = response.json()
        
        if data['status'] == 'ok' and data['articles']:
            df = pd.DataFrame(data['articles'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            df = df.sort_values('publishedAt', ascending=False).reset_index(drop=True)
            
            print(f"  ✅ Success: {len(df)} articles")
            
            # Save news data
            os.makedirs("data/raw", exist_ok=True)
            filename = "data/raw/news_data.csv"
            df.to_csv(filename, index=False)
            print(f"  💾 Saved: {filename}")
            
            # Show sample headlines
            print("  📄 Sample headlines:")
            for i, title in enumerate(df['title'].head(3)):
                print(f"    {i+1}. {title[:60]}...")
            
            return df
        else:
            print(f"  ❌ Error: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return pd.DataFrame()

def main():
    """Main data collection function."""
    print("🚀 Neural Market Predictor - Data Collection")
    print("=" * 60)
    print("👤 User: Utkarsh-upadhyay9")
    print(f"⏰ Time: {datetime.now()}")
    print("🔑 Using: Alpha Vantage + NewsAPI + Yahoo Finance")
    
    # Define symbols to collect
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]
    
    print(f"\n📋 Target symbols: {symbols}")
    
    # Collect Yahoo Finance data (unlimited, reliable)
    yahoo_data = collect_yahoo_data(symbols)
    
    # Collect Alpha Vantage data (limited by free tier)
    alpha_data = collect_alpha_vantage_data(symbols[:3])  # Limit to 3 symbols
    
    # Collect news data
    news_data = collect_news_data("financial markets technology stocks")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COLLECTION SUMMARY")
    print(f"✅ Yahoo Finance: {len(yahoo_data)} symbols")
    print(f"✅ Alpha Vantage: {len(alpha_data)} symbols")
    print(f"✅ News Articles: {len(news_data)} articles")
    
    # Show file structure
    print(f"\n📁 Files created in data/raw/:")
    try:
        files = os.listdir("data/raw")
        for file in sorted(files):
            print(f"  📄 {file}")
    except:
        print("  (Directory not found)")
    
    print(f"\n🎉 Data collection completed!")
    print(f"💡 Next steps:")
    print(f"  1. Run preprocessing: python scripts/preprocess_data.py")
    print(f"  2. Train models: python scripts/train_model.py")
    print(f"  3. Launch dashboard: streamlit run src/visualization/dashboard.py")

if __name__ == "__main__":
    main()
