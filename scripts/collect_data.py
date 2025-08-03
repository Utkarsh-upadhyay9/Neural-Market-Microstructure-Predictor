#!/usr/bin/env python3
"""
Enhanced data collection script with all available APIs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collector import DataCollector
from utils import setup_logging, save_json
import pandas as pd
from datetime import datetime
import argparse


def main():
    """Main function to collect market data."""
    parser = argparse.ArgumentParser(description='Collect market data')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL'], 
                       help='Stock symbols to collect')
    parser.add_argument('--source', choices=['yahoo', 'alpha_vantage', 'nasdaq'], 
                       default='yahoo', help='Data source')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--include-news', action='store_true',
                       help='Include news data collection')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize data collector
        collector = DataCollector(args.config)
        
        print(f" Collecting data for symbols: {args.symbols}")
        print(f" Using source: {args.source}")
        print(f" User: Utkarsh-upadhyay9")
        print(f"⏰ Time: {datetime.now()}")
        
        # Collect market data
        if args.source == 'yahoo':
            for symbol in args.symbols:
                print(f"\n Fetching Yahoo data for {symbol}...")
                data = collector.get_yahoo_data(symbol, period="1y")
                
                if not data.empty:
                    print(f" Success: {len(data)} records")
                    filename = f"data/raw/{symbol}_yahoo_1y.csv"
                    collector.save_data(data, filename)
                    print(f" Saved to: {filename}")
                else:
                    print(f" Failed for {symbol}")
        
        elif args.source == 'alpha_vantage':
            for symbol in args.symbols:
                print(f"\n Fetching Alpha Vantage data for {symbol}...")
                
                # Get daily data first (uses fewer API calls)
                daily_data = collector.get_alpha_vantage_data(symbol, interval="daily")
                if not daily_data.empty:
                    print(f" Daily data: {len(daily_data)} records")
                    filename = f"data/raw/{symbol}_alphavantage_daily.csv"
                    collector.save_data(daily_data, filename)
                
                # Wait to respect rate limits
                print("⏳ Waiting for rate limit...")
                import time
                time.sleep(12)  # Alpha Vantage: 5 calls per minute
        
        elif args.source == 'nasdaq':
            for symbol in args.symbols:
                print(f"\n Fetching Nasdaq data for {symbol}...")
                
                # Try different dataset codes
                dataset_codes = [f"WIKI/{symbol}", f"EOD/{symbol}"]
                
                for dataset_code in dataset_codes:
                    data = collector.get_nasdaq_data(dataset_code, symbol)
                    if not data.empty:
                        print(f" Success with {dataset_code}: {len(data)} records")
                        filename = f"data/raw/{symbol}_nasdaq.csv"
                        collector.save_data(data, filename)
                        break
                else:
                    print(f" No data found for {symbol}")
        
        # Collect news data if requested
        if args.include_news:
            print(f"\n📰 Collecting news data...")
            
            # NewsAPI
            news_data = collector.get_news_data("stock market financial", days_back=7)
            if not news_data.empty:
                print(f" NewsAPI: {len(news_data)} articles")
                filename = "data/raw/news_general.csv"
                collector.save_data(news_data, filename)
            
            # Alpha Vantage News
            av_news = collector.get_alpha_vantage_news("financial_markets")
            if not av_news.empty:
                print(f" Alpha Vantage News: {len(av_news)} articles")
                filename = "data/raw/news_alphavantage.csv"
                collector.save_data(av_news, filename)
        
        # Test real-time data
        print(f"\n🔴 Testing real-time data for {args.symbols[0]}...")
        real_time = collector.get_real_time_data(args.symbols[0])
        if real_time:
            print(" Real-time data available:")
            for key, value in real_time.items():
                if key != 'timestamp':
                    print(f"  {key}: {value}")
        
        print(f"\n Data collection completed!")
        print(f" Check 'data/raw/' directory for files")
        
    except Exception as e:
        print(f" Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
