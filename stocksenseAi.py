import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from textblob import TextBlob
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NSE Pro Screener & Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .volume-spike { 
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 100%);
        color: #d63384;
        font-weight: bold;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .dii-increase { color: #28a745; font-weight: bold; }
    .dii-decrease { color: #dc3545; font-weight: bold; }
    .fii-increase { color: #17a2b8; font-weight: bold; }
    .fii-decrease { color: #fd7e14; font-weight: bold; }
    .promoter-increase { color: #6f42c1; font-weight: bold; }
    .promoter-decrease { color: #e83e8c; font-weight: bold; }
    .stSpinner > div {
        border-top-color: #667eea;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #764ba2;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.5);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with enhanced data structures
def initialize_session_state():
    defaults = {
        'stock_data': pd.DataFrame(),
        'technical_data': {}, # Will store more detailed technical data for charts
        'news_sentiment': {}, # Will store more detailed sentiment for charts
        'volume_alerts': [],
        'stakeholder_changes': {},
        'last_update': None,
        'auto_update_enabled': False,
        'alert_settings': {
            'volume_spike_threshold': 200,  # 200% above average
            'stakeholder_change_threshold': 1.0  # 1% change
        },
        'selected_stock_for_chart': None,
        'trigger_update': False # Flag to trigger manual/auto update
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

class EnhancedNSEScreener:
    def __init__(self):
        self.nse_base_url = "https://www.nseindia.com/api"
        self.finology_base_url = "https://api.finology.in/v1"
        self.news_api_key = st.secrets.get("NEWS_API_KEY", "your_news_api_key_here") # Get from Streamlit secrets
        self.finology_api_key = st.secrets.get("FINOLOGY_API_KEY", "your_finology_api_key_here") # Get from Streamlit secrets
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    @st.cache_data(ttl=3600) # Cache for 1 hour
    def get_nse_top_stocks(self, count=200):
        """Get top NSE stocks from multiple indices"""
        try:
            # Fetch from NIFTY 500 for comprehensive coverage
            indices = ['NIFTY 50', 'NIFTY NEXT 50', 'NIFTY MIDCAP 100', 'NIFTY SMALLCAP 100', 'NIFTY 500']
            all_stocks = set()

            for index in indices:
                try:
                    url = f"{self.nse_base_url}/equity-stockIndices?index={index.replace(' ', '%20')}"
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        stocks = [stock['symbol'] + '.NS' for stock in data.get('data', [])]
                        all_stocks.update(stocks)
                except Exception as e:
                    # st.warning(f"Error fetching {index}: {str(e)}")
                    continue

            # Fallback to comprehensive stock list if API fails
            if not all_stocks:
                all_stocks = self._get_fallback_stocks()

            return sorted(list(all_stocks))[:count]

        except Exception as e:
            st.error(f"Error fetching NSE stocks: {str(e)}")
            return sorted(self._get_fallback_stocks())[:count]

    def _get_fallback_stocks(self):
        """Comprehensive fallback stock list"""
        return [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
            'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'POWERGRID.NS',
            'NTPC.NS', 'TECHM.NS', 'HCLTECH.NS', 'BAJFINANCE.NS', 'COALINDIA.NS',
            'ONGC.NS', 'TATASTEEL.NS', 'GRASIM.NS', 'ADANIPORTS.NS', 'BRITANNIA.NS',
            'DRREDDY.NS', 'EICHERMOT.NS', 'BAJAJFINSV.NS', 'CIPLA.NS', 'HEROMOTOCO.NS',
            'INDUSINDBK.NS', 'APOLLOHOSP.NS', 'DIVISLAB.NS', 'JSWSTEEL.NS', 'M&M.NS',
            'TATAMOTORS.NS', 'HINDALCO.NS', 'ADANIENT.NS', 'HDFCLIFE.NS', 'SBILIFE.NS',
            'BAJAJ-AUTO.NS', 'GODREJCP.NS', 'DABUR.NS', 'VEDL.NS', 'BANKBARODA.NS',
            'PIDILITIND.NS', 'BERGEPAINT.NS', 'HAVELLS.NS', 'PAGEIND.NS', 'BOSCHLTD.NS',
            'DMART.NS', 'PIIND.NS', 'MPHASIS.NS', 'LTIM.NS', 'ZYDUSLIFE.NS', 'POLYCAB.NS',
            'SRF.NS', 'SUPREMEIND.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS', 'CAMS.NS'
        ]

    async def fetch_comprehensive_data(self, symbols):
        """Async fetch for multiple data sources"""
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                tasks.append(self._fetch_single_stock_data(session, symbol))

            # Process in batches to avoid rate limiting
            batch_size = 10
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, dict) and result: # Ensure it's a valid dict and not empty
                        results.append(result)

                # Rate limiting between batches
                await asyncio.sleep(0.5) # Reduced sleep for faster loading

        return pd.DataFrame(results)

    async def _fetch_single_stock_data(self, session, symbol):
        """Fetch comprehensive data for a single stock"""
        try:
            # Parallel data fetching
            tasks = [
                self._fetch_basic_data(session, symbol),
                self._fetch_technical_data(symbol), # yfinance is not async with aiohttp
                self._fetch_volume_data(symbol), # yfinance is not async with aiohttp
                self._fetch_stakeholder_data(session, symbol),
                self._fetch_news_sentiment(session, symbol)
            ]

            basic_data, technical_data, volume_data, stakeholder_data, sentiment = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Combine all data
            combined_data = {**basic_data} if isinstance(basic_data, dict) else {}

            if isinstance(technical_data, dict):
                combined_data.update(technical_data)
            if isinstance(volume_data, dict):
                combined_data.update(volume_data)
            if isinstance(stakeholder_data, dict):
                combined_data.update(stakeholder_data)
            if isinstance(sentiment, dict):
                combined_data.update(sentiment)

            # Ensure 'Symbol' is always present
            if 'Symbol' not in combined_data or not combined_data['Symbol']:
                combined_data['Symbol'] = symbol

            # Calculate enhanced recommendation score
            combined_data['Enhanced_Score'] = self._calculate_enhanced_score(combined_data)
            combined_data['Risk_Level'] = self._calculate_risk_level(combined_data)

            return combined_data

        except Exception as e:
            # st.warning(f"Error fetching data for {symbol}: {str(e)}")
            return {'Symbol': symbol} # Return at least symbol to indicate attempt

    async def _fetch_basic_data(self, session, symbol):
        """Fetch basic stock data from multiple sources"""
        try:
            # Try NSE API first, fallback to Finology, then yfinance
            data = await self._try_nse_api(session, symbol)
            if not data:
                data = await self._try_finology_api(session, symbol)
            if not data:
                # Use a separate thread for yfinance as it's blocking
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, self._try_yfinance_fallback_sync, symbol)

            return data
        except:
            return {}

    async def _try_nse_api(self, session, symbol):
        """Try NSE API for stock data"""
        try:
            clean_symbol = symbol.replace('.NS', '')
            url = f"{self.nse_base_url}/quote-equity?symbol={clean_symbol}"

            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_nse_data(data, symbol)
        except:
            pass
        return None

    async def _try_finology_api(self, session, symbol):
        """Try Finology API for stock data"""
        try:
            clean_symbol = symbol.replace('.NS', '')
            url = f"{self.finology_base_url}/equity/info/{clean_symbol}"
            headers = {'X-API-Key': self.finology_api_key}

            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_finology_data(data, symbol)
        except:
            pass
        return None

    def _try_yfinance_fallback_sync(self, symbol):
        """Fallback to yfinance for stock data (synchronous)"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Fetch less historical data for basic info, more for technicals
            hist = ticker.history(period="1mo")

            if hist.empty:
                return None

            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            change_percent = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100 if len(hist) > 1 else 0

            return {
                'Symbol': symbol,
                'Name': info.get('longName', symbol.replace('.NS', '')),
                'Current_Price': round(current_price, 2),
                'Volume': volume,
                'Market_Cap': info.get('marketCap', 0),
                'PE_Ratio': info.get('trailingPE', 0),
                'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'Debt_Equity': info.get('debtToEquity', 0),
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                'Change_Percent': round(change_percent, 2),
                'High_52W': info.get('fiftyTwoWeekHigh', 0),
                'Low_52W': info.get('fiftyTwoWeekLow', 0)
            }
        except Exception as e:
            # print(f"yfinance fallback failed for {symbol}: {e}")
            return None

    def _parse_nse_data(self, data, symbol):
        """Parse NSE API response"""
        try:
            price_info = data.get('priceInfo', {})
            return {
                'Symbol': symbol,
                'Name': data.get('info', {}).get('companyName', symbol.replace('.NS', '')),
                'Current_Price': price_info.get('lastPrice', 0),
                'Volume': data.get('marketDeptOrderBook', {}).get('totalTradedVolume', 0),
                'Market_Cap': data.get('info', {}).get('marketCap', 0), # NSE API doesn't always provide MCap directly
                'Change_Percent': price_info.get('pChange', 0),
                'High_52W': price_info.get('weekHighLow', {}).get('max', 0),
                'Low_52W': price_info.get('weekHighLow', {}).get('min', 0)
            }
        except:
            return {}

    def _parse_finology_data(self, data, symbol):
        """Parse Finology API response"""
        try:
            return {
                'Symbol': symbol,
                'Name': data.get('name', symbol.replace('.NS', '')),
                'Current_Price': data.get('price', 0),
                'PE_Ratio': data.get('pe', 0),
                'ROE': data.get('roe', 0),
                'Debt_Equity': data.get('debt_equity', 0),
                'Revenue_Growth': data.get('revenue_growth', 0),
                'Profit_Growth': data.get('profit_growth', 0),
                'Market_Cap': data.get('market_cap', 0) # Finology provides Market Cap
            }
        except:
            return {}

    # Synchronous function for yfinance, to be called in a thread pool
    def _fetch_technical_data(self, symbol):
        """Fetch technical indicators using yfinance (synchronous)"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y") # Get 1 year data for MA/RSI calculations

            if len(hist) < 50:
                return {} # Not enough data for meaningful indicators

            close_prices = hist['Close'].values
            high_prices = hist['High'].values
            low_prices = hist['Low'].values

            # Calculate technical indicators
            rsi = talib.RSI(close_prices, timeperiod=14)[-1]
            macd, macd_signal, macd_hist = talib.MACD(close_prices)

            sma_20 = talib.SMA(close_prices, timeperiod=20)[-1]
            sma_50 = talib.SMA(close_prices, timeperiod=50)[-1]
            ema_12 = talib.EMA(close_prices, timeperiod=12)[-1]
            ema_26 = talib.EMA(close_prices, timeperiod=26)[-1]

            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(close_prices)

            # Support and Resistance levels (simple min/max for recent period)
            support = np.min(low_prices[-20:])
            resistance = np.max(high_prices[-20:])

            # Store full history for charting
            st.session_state.technical_data[symbol] = {
                'history': hist.reset_index().rename(columns={'index': 'Date'}),
                'RSI': talib.RSI(close_prices, timeperiod=14),
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Hist': macd_hist,
                'SMA_20': talib.SMA(close_prices, timeperiod=20),
                'SMA_50': talib.SMA(close_prices, timeperiod=50),
                'Bollinger_Upper': bollinger_upper,
                'Bollinger_Middle': bollinger_middle,
                'Bollinger_Lower': bollinger_lower
            }

            return {
                'RSI': round(rsi, 2) if not np.isnan(rsi) else 50,
                'MACD': round(macd[-1], 4) if not np.isnan(macd[-1]) else 0,
                'MACD_Signal': round(macd_signal[-1], 4) if not np.isnan(macd_signal[-1]) else 0,
                'SMA_20': round(sma_20, 2) if not np.isnan(sma_20) else 0,
                'SMA_50': round(sma_50, 2) if not np.isnan(sma_50) else 0,
                'EMA_12': round(ema_12, 2) if not np.isnan(ema_12) else 0,
                'EMA_26': round(ema_26, 2) if not np.isnan(ema_26) else 0,
                'Bollinger_Upper': round(bollinger_upper[-1], 2) if not np.isnan(bollinger_upper[-1]) else 0,
                'Bollinger_Lower': round(bollinger_lower[-1], 2) if not np.isnan(bollinger_lower[-1]) else 0,
                'Support': round(support, 2),
                'Resistance': round(resistance, 2),
                'Technical_Signal': self._get_technical_signal(rsi, macd[-1], macd_signal[-1], close_prices[-1], sma_20, sma_50)
            }
        except Exception as e:
            # print(f"Technical data fetch failed for {symbol}: {e}")
            return {}

    def _get_technical_signal(self, rsi, macd, macd_signal, current_price, sma_20, sma_50):
        """Generate technical trading signal"""
        signals = []

        # RSI signals
        if rsi < 30:
            signals.append("Oversold (Buy)")
        elif rsi > 70:
            signals.append("Overbought (Sell)")

        # MACD signals
        if macd > macd_signal and macd_hist > 0: # Check MACD histogram for strength
            signals.append("MACD Bullish Cross")
        elif macd < macd_signal and macd_hist < 0:
            signals.append("MACD Bearish Cross")

        # Moving average signals (Golden Cross / Death Cross)
        if sma_20 > sma_50 and current_price > sma_20:
            signals.append("MA Bullish (Strong)")
        elif sma_50 > sma_20 and current_price < sma_20:
            signals.append("MA Bearish (Strong)")
        elif current_price > sma_20:
            signals.append("Above SMA 20")
        elif current_price < sma_20:
            signals.append("Below SMA 20")


        return " | ".join(signals) if signals else "Neutral"

    # Synchronous function for yfinance, to be called in a thread pool
    def _fetch_volume_data(self, symbol):
        """Fetch volume data and detect spikes (synchronous)"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d") # Get 60 days for 20-day average

            if len(hist) < 20:
                return {}

            current_volume = hist['Volume'].iloc[-1]
            avg_volume_20d = hist['Volume'][-20:-1].mean() # Average excluding today's volume
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1

            # Detect volume spike
            volume_spike = volume_ratio > (st.session_state.alert_settings['volume_spike_threshold'] / 100)

            return {
                'Current_Volume': int(current_volume),
                'Avg_Volume_20D': int(avg_volume_20d),
                'Volume_Ratio': round(volume_ratio, 2),
                'Volume_Spike': volume_spike,
                'Volume_Status': 'High' if volume_ratio >= 1.5 else 'Normal' if volume_ratio >= 0.8 else 'Low'
            }
        except Exception as e:
            # print(f"Volume data fetch failed for {symbol}: {e}")
            return {}

    async def _fetch_stakeholder_data(self, session, symbol):
        """Fetch DII, FII, and Promoter holding data (simulated)"""
        try:
            # In a real application, you would integrate with APIs like:
            # - NSE Corporate Actions API (for shareholding changes)
            # - BSE Shareholding Pattern API
            # - Paid data providers (e.g., Trendlyne, Screener.in APIs)
            
            # --- Simulated Data for Demonstration ---
            clean_symbol = symbol.replace('.NS', '')
            
            # Use symbol hash for consistent 'randomness'
            random_seed = hash(clean_symbol) % (2**32 - 1)
            np.random.seed(random_seed)

            # Base holdings - make them somewhat realistic for different companies
            base_promoter = np.random.uniform(20, 75)
            base_dii = np.random.uniform(5, 30)
            base_fii = np.random.uniform(5, 40)
            
            # Changes are relative to base
            promoter_change = np.random.uniform(-2.0, 2.0) # +/- 2%
            dii_change = np.random.uniform(-1.5, 1.5)
            fii_change = np.random.uniform(-2.5, 2.5)

            # Apply changes for 'current' values, ensuring bounds
            current_promoter = max(0, min(100, base_promoter + promoter_change))
            current_dii = max(0, min(100, base_dii + dii_change))
            current_fii = max(0, min(100, base_fii + fii_change))
            
            # Ensure total doesn't exceed 100 significantly, though individual can.
            # This is simplified; real data has complex rules.
            
            stakeholder_alert = abs(promoter_change) > st.session_state.alert_settings['stakeholder_change_threshold'] or \
                                   abs(dii_change) > st.session_state.alert_settings['stakeholder_change_threshold'] or \
                                   abs(fii_change) > st.session_state.alert_settings['stakeholder_change_threshold']

            return {
                'Promoter_Holding': round(current_promoter, 2),
                'DII_Holding': round(current_dii, 2),
                'FII_Holding': round(current_fii, 2),
                'Promoter_Change': round(promoter_change, 2),
                'DII_Change': round(dii_change, 2),
                'FII_Change': round(fii_change, 2),
                'Stakeholder_Alert': stakeholder_alert
            }
        except Exception as e:
            # print(f"Stakeholder data fetch failed for {symbol}: {e}")
            return {}

    async def _fetch_news_sentiment(self, session, symbol):
        """Fetch and analyze news sentiment"""
        try:
            clean_symbol = symbol.replace('.NS', '')

            # Multiple news sources (using placeholders, replace with actual)
            # You might need to find specific APIs for Indian stock news or use broad news APIs
            news_sources = [
                f"https://newsapi.org/v2/everything?q={clean_symbol} stock india&language=en&sortBy=publishedAt&pageSize=10&apiKey={self.news_api_key}",
                # f"https://api.marketaux.com/v1/news/all?symbols={clean_symbol}&filter_entities=true&language=en&api_token={self.news_api_key}" # Example, if you have Marketaux
            ]

            all_headlines = []

            for url in news_sources:
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            data = await response.json()

                            if 'newsapi' in url:
                                headlines = [article['title'] for article in data.get('articles', []) if article.get('title')]
                            # elif 'marketaux' in url:
                            #     headlines = [article['title'] for article in data.get('data', []) if article.get('title')]
                            else:
                                headlines = []

                            all_headlines.extend(headlines[:5])  # Limit to 5 per source
                except Exception as e:
                    # print(f"News API fetch failed for {url}: {e}")
                    continue

            if not all_headlines:
                # Fallback to simulated sentiment if no news found
                sentiment_score = np.random.uniform(-0.4, 0.4) # Slightly less extreme for simulation
                return {
                    'News_Sentiment': round(sentiment_score, 3),
                    'Sentiment_Label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral',
                    'News_Count': 0,
                    'Latest_Headlines': ["No recent news found."]
                }

            # Analyze sentiment using TextBlob
            total_sentiment = 0
            for headline in all_headlines:
                blob = TextBlob(headline)
                total_sentiment += blob.sentiment.polarity

            avg_sentiment = total_sentiment / len(all_headlines)

            # Store all headlines for detailed view
            st.session_state.news_sentiment[symbol] = {
                'headlines': all_headlines,
                'sentiment_score': avg_sentiment
            }

            return {
                'News_Sentiment': round(avg_sentiment, 3),
                'Sentiment_Label': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral',
                'News_Count': len(all_headlines),
                'Latest_Headlines': all_headlines[:3]  # Store top 3 headlines for table display
            }

        except Exception as e:
            # print(f"News sentiment analysis failed for {symbol}: {e}")
            return {}

    def _calculate_enhanced_score(self, data):
        """Enhanced scoring algorithm with technical and sentiment factors"""
        score = 0
        
        # Fundamental scoring (40% weight)
        pe = data.get('PE_Ratio', 0)
        roe = data.get('ROE', 0)
        debt_equity = data.get('Debt_Equity', 0)
        
        # PE Ratio (Lower is generally better, but not negative)
        if pe > 0:
            if pe < 15: score += 15
            elif pe < 25: score += 10
            elif pe < 35: score += 5
            else: score += 0 # Very high PE might indicate overvaluation
        
        # ROE (Higher is better)
        if roe > 20: score += 15
        elif roe > 15: score += 12
        elif roe > 10: score += 8
        elif roe > 5: score += 5
        
        # Debt to Equity (Lower is better)
        if debt_equity < 0.5: score += 10
        elif debt_equity < 1: score += 7
        elif debt_equity < 2: score += 3
        
        # Technical scoring (30% weight)
        rsi = data.get('RSI', 50)
        # RSI between 30-70 is neutral, <30 oversold (potential buy), >70 overbought (potential sell)
        if 30 < rsi < 70: score += 10  # Healthy range
        elif rsi < 30: score += 15  # Oversold - potential buy signal
        elif rsi > 70: score -= 5 # Overbought - potential sell signal
        
        macd = data.get('MACD', 0)
        macd_signal = data.get('MACD_Signal', 0)
        if macd > macd_signal: score += 10 # Bullish MACD crossover
        elif macd < macd_signal: score -= 5 # Bearish MACD crossover
        
        current_price = data.get('Current_Price', 0)
        sma_20 = data.get('SMA_20', 0)
        sma_50 = data.get('SMA_50', 0)
        
        if current_price > sma_20 and sma_20 > 0: score += 8 # Price above short-term MA
        if current_price > sma_50 and sma_50 > 0: score += 7 # Price above medium-term MA
        if sma_20 > sma_50 and sma_50 > 0: score += 5 # Golden Cross (short MA > long MA)

        # Volume scoring (15% weight)
        volume_ratio = data.get('Volume_Ratio', 1)
        if volume_ratio > 2.0: score += 10  # Very high volume spike
        elif volume_ratio > 1.5: score += 7 # High volume
        elif volume_ratio > 1.0: score += 3 # Above average volume
        
        # Sentiment scoring (15% weight)
        sentiment = data.get('News_Sentiment', 0)
        if sentiment > 0.2: score += 8 # Strong positive sentiment
        elif sentiment > 0.05: score += 5 # Moderate positive sentiment
        elif sentiment < -0.2: score -= 8 # Strong negative sentiment
        elif sentiment < -0.05: score -= 3 # Moderate negative sentiment
        
        return int(max(0, min(score, 100))) # Cap score between 0 and 100

    def _calculate_risk_level(self, data):
        """Calculate risk level based on multiple factors"""
        risk_score = 0
        
        # Volatility risk (RSI extreme, price far from MAs, high beta - not included here)
        rsi = data.get('RSI', 50)
        if rsi > 80 or rsi < 20:
            risk_score += 2 # Extreme RSI indicates potential reversal/high volatility

        current_price = data.get('Current_Price', 0)
        sma_20 = data.get('SMA_20', 0)
        if sma_20 > 0 and abs(current_price - sma_20) / sma_20 > 0.10: # Price 10% away from SMA 20
             risk_score += 1
        
        # Debt risk
        if data.get('Debt_Equity', 0) > 2.0:
            risk_score += 3
        elif data.get('Debt_Equity', 0) > 1.0:
            risk_score += 1
        
        # Volume risk (unusual volume without clear direction)
        volume_ratio = data.get('Volume_Ratio', 1)
        if volume_ratio > 3.0: # Very high unusual volume
            risk_score += 2
        
        # Sentiment risk
        sentiment = data.get('News_Sentiment', 0)
        if sentiment < -0.2:
            risk_score += 2 # Negative sentiment
        
        # Stakeholder risk (significant reduction)
        if data.get('Promoter_Change', 0) < -1.5:
            risk_score += 3
        if data.get('DII_Change', 0) < -1.0:
            risk_score += 1
        if data.get('FII_Change', 0) < -1.0:
            risk_score += 1
            
        if risk_score >= 6:
            return "High Risk"
        elif risk_score >= 3:
            return "Medium Risk"
        else:
            return "Low Risk"

# Auto-update scheduler
def schedule_updates():
    """Schedule automatic updates"""
    def run_update():
        if st.session_state.get('auto_update_enabled', False):
            st.session_state.trigger_update = True
            # Streamlit rerun is needed to pick up session state changes
            # This will trigger a rerun in the main thread
            st.experimental_rerun()
            
    # Schedule to run every 15 minutes, but only when Streamlit is active
    # The actual execution will depend on the main thread checking 'trigger_update'
    schedule.every(15).minutes.do(run_update)
    
    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute

# Start scheduler in background if not already started
if 'scheduler_started' not in st.session_state:
    st.session_state.scheduler_started = True
    scheduler_thread = threading.Thread(target=schedule_updates, daemon=True)
    scheduler_thread.start()

# --- Main Streamlit App Functions ---

def display_alerts():
    st.header("🚨 Real-time Alerts")
    
    volume_alerts = [stock for _, stock in st.session_state.stock_data.iterrows() if stock.get('Volume_Spike', False)]
    stakeholder_alerts = [stock for _, stock in st.session_state.stock_data.iterrows() if stock.get('Stakeholder_Alert', False)]
    
    st.markdown("### Volume Spikes")
    if volume_alerts:
        for alert in volume_alerts:
            st.markdown(f"""
            <div class="alert-card">
                <h5>📈 Volume Spike Alert: {alert['Symbol']} - {alert.get('Name', '')}</h5>
                <p>Current Volume: {alert.get('Current_Volume', 'N/A'):,} | 20-Day Avg: {alert.get('Avg_Volume_20D', 'N/A'):,} | Ratio: <span class="volume-spike">{alert.get('Volume_Ratio', 'N/A')}x</span></p>
                <p>Possible significant price movement. Current Price: {alert.get('Current_Price', 'N/A')} ({alert.get('Change_Percent', 'N/A'):.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant volume spikes detected.")
        
    st.markdown("### Significant Stakeholder Changes")
    if stakeholder_alerts:
        for alert in stakeholder_alerts:
            st.markdown(f"""
            <div class="alert-card">
                <h5>👥 Stakeholder Change Alert: {alert['Symbol']} - {alert.get('Name', '')}</h5>
                <p>
                    Promoter: {alert.get('Promoter_Holding', 'N/A')}% ({alert.get('Promoter_Change', 'N/A'):+.2f}%) |
                    DII: {alert.get('DII_Holding', 'N/A')}% ({alert.get('DII_Change', 'N/A'):+.2f}%) |
                    FII: {alert.get('FII_Holding', 'N/A')}% ({alert.get('FII_Change', 'N/A'):+.2f}%)
                </p>
                <p>Monitor for potential impact on stock price and governance.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No significant stakeholder changes detected.")

def display_technical_analysis():
    st.header("📈 Technical Analysis")

    if st.session_state.stock_data.empty:
        st.warning("Please fetch stock data first in the 'Enhanced Screener' tab.")
        return

    symbols = st.session_state.stock_data['Symbol'].tolist()
    
    selected_symbol = st.selectbox("Select Stock for Detailed Technical Chart:", symbols, 
                                   key='ta_select_stock')
    
    if selected_symbol and selected_symbol in st.session_state.technical_data:
        tech_data = st.session_state.technical_data[selected_symbol]
        df_hist = tech_data['history']

        if df_hist.empty:
            st.warning(f"No historical data available for {selected_symbol}.")
            return
        
        st.subheader(f"Candlestick Chart with Indicators for {selected_symbol.replace('.NS', '')}")

        # Create subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.08,
                            row_heights=[0.6, 0.2, 0.2]) # Candlestick, Volume, RSI

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df_hist['Date'],
                                     open=df_hist['Open'],
                                     high=df_hist['High'],
                                     low=df_hist['Low'],
                                     close=df_hist['Close'],
                                     name='Candlesticks'), row=1, col=1)

        # Moving Averages
        if 'SMA_20' in tech_data and len(tech_data['SMA_20']) == len(df_hist):
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['SMA_20'], mode='lines',
                                     name='SMA 20', line=dict(color='orange', width=1)), row=1, col=1)
        if 'SMA_50' in tech_data and len(tech_data['SMA_50']) == len(df_hist):
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['SMA_50'], mode='lines',
                                     name='SMA 50', line=dict(color='purple', width=1)), row=1, col=1)

        # Bollinger Bands
        if 'Bollinger_Upper' in tech_data and len(tech_data['Bollinger_Upper']) == len(df_hist):
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['Bollinger_Upper'], mode='lines',
                                     name='Upper BB', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['Bollinger_Middle'], mode='lines',
                                     name='Middle BB', line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['Bollinger_Lower'], mode='lines',
                                     name='Lower BB', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

        # Volume chart
        colors = ['green' if df_hist.loc[i, 'Close'] >= df_hist.loc[i, 'Open'] else 'red' for i in df_hist.index]
        fig.add_trace(go.Bar(x=df_hist['Date'], y=df_hist['Volume'], name='Volume', marker_color=colors), row=2, col=1)

        # RSI chart
        if 'RSI' in tech_data and len(tech_data['RSI']) == len(df_hist):
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['RSI'], mode='lines',
                                     name='RSI', line=dict(color='teal')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="grey", row=3, col=1)

        fig.update_layout(
            title_text=f"{selected_symbol.replace('.NS', '')} Technicals",
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode="x unified",
            template="plotly_dark",
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Update Y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display MACD separately as it often needs a second y-axis or separate chart for clarity
        st.subheader("MACD Chart")
        if 'MACD' in tech_data and len(tech_data['MACD']) == len(df_hist):
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df_hist['Date'], y=tech_data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
            
            # MACD Histogram
            macd_hist_colors = ['green' if val >= 0 else 'red' for val in tech_data['MACD_Hist']]
            fig_macd.add_trace(go.Bar(x=df_hist['Date'], y=tech_data['MACD_Hist'], name='Histogram', marker_color=macd_hist_colors, opacity=0.7))
            
            fig_macd.update_layout(
                title_text="MACD Indicator",
                xaxis_rangeslider_visible=False,
                height=350,
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig_macd, use_container_width=True)

    else:
        st.info("No detailed technical data available for the selected stock. Fetching data usually populates this section.")

def display_smart_portfolio():
    st.header("💼 Smart Portfolio Analysis (Coming Soon!)")
    st.write("This section will allow you to import your portfolio, get real-time performance insights, diversification analysis, and personalized recommendations based on the screener data.")
    st.info("Features envisioned:")
    st.markdown("- **Portfolio Performance:** Track daily P&L, overall gains/losses.")
    st.markdown("- **Diversification Matrix:** Analyze sector, industry, and market cap allocation.")
    st.markdown("- **Risk Assessment:** Evaluate portfolio's overall risk level based on constituent stocks.")
    st.markdown("- **Rebalancing Suggestions:** Get alerts for portfolio imbalances and rebalancing opportunities.")
    st.markdown("- **Watchlist Integration:** Add stocks from the screener directly to your watchlist.")

def display_sentiment_analysis():
    st.header("📰 News and Sentiment Analysis")

    if st.session_state.stock_data.empty:
        st.warning("Please fetch stock data first in the 'Enhanced Screener' tab.")
        return

    symbols = st.session_state.stock_data['Symbol'].tolist()
    selected_symbol = st.selectbox("Select Stock for News Sentiment:", symbols, key='sentiment_select_stock')

    if selected_symbol and selected_symbol in st.session_state.news_sentiment:
        sentiment_data = st.session_state.news_sentiment[selected_symbol]
        
        st.subheader(f"Sentiment for {selected_symbol.replace('.NS', '')}")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            sentiment_label = 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
            st.metric("Overall Sentiment Score", f"{sentiment_score:.3f}", 
                      delta_color="normal" if sentiment_label == 'Neutral' else ("inverse" if sentiment_label == 'Negative' else "normal"))
        with col_s2:
            st.metric("Sentiment Label", sentiment_label)
        
        st.markdown("---")
        st.subheader("Latest Headlines")
        headlines = sentiment_data.get('headlines', [])
        if headlines:
            for i, headline in enumerate(headlines):
                st.markdown(f"**{i+1}.** {headline}")
        else:
            st.info("No recent headlines found for this stock.")
    else:
        st.info("No detailed news sentiment data available for the selected stock. Data fetching usually populates this section.")

def display_stakeholder_tracking():
    st.header("👥 Stakeholder Tracking")

    if st.session_state.stock_data.empty:
        st.warning("Please fetch stock data first in the 'Enhanced Screener' tab.")
        return

    st.write("Track changes in Promoter, DII (Domestic Institutional Investors), and FII (Foreign Institutional Investors) holdings.")

    # Filter for stocks with stakeholder data
    stakeholder_df = st.session_state.stock_data[[
        'Symbol', 'Name', 'Promoter_Holding', 'Promoter_Change',
        'DII_Holding', 'DII_Change', 'FII_Holding', 'FII_Change'
    ]].dropna(subset=['Promoter_Holding'])

    if stakeholder_df.empty:
        st.info("No stakeholder data available for display yet.")
        return

    # Add color formatting based on change
    def color_change(val):
        try:
            val = float(val)
            if val > 0:
                return 'background-color: #d4edda; color: #155724' # Light green
            elif val < 0:
                return 'background-color: #f8d7da; color: #721c24' # Light red
            else:
                return ''
        except:
            return ''

    st.dataframe(
        stakeholder_df.style.applymap(
            color_change, subset=['Promoter_Change', 'DII_Change', 'FII_Change']
        ).format({
            'Promoter_Holding': "{:.2f}%", 'DII_Holding': "{:.2f}%", 'FII_Holding': "{:.2f}%",
            'Promoter_Change': "{:+.2f}%", 'DII_Change': "{:+.2f}%", 'FII_Change': "{:+.2f}%"
        }),
        use_container_width=True
    )
    st.markdown("---")
    st.info("Note: Stakeholder data is simulated for demonstration purposes. Real data requires specialized APIs.")


def display_settings():
    st.header("⚙️ Application Settings")
    st.write("Configure various parameters for the screener and alerts.")
    
    st.subheader("API Keys (for full functionality)")
    st.warning("Ensure these API keys are set up in Streamlit Secrets (`.streamlit/secrets.toml`) for production use. Replace 'your_...' placeholders with actual keys.")
    st.code("""
# .streamlit/secrets.toml
NEWS_API_KEY = "your_news_api_key_here"
FINOLOGY_API_KEY = "your_finology_api_key_here"
    """, language="toml")
    
    st.subheader("Data Refresh Options")
    st.write(f"Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'N/A'}")
    
    st.subheader("Alert Configuration")
    
    st.write("Adjust the sensitivity of real-time alerts:")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.session_state.alert_settings['volume_spike_threshold'] = st.slider(
            "Volume Spike Threshold (current volume % above 20-day avg)",
            min_value=100, max_value=500, step=50,
            value=st.session_state.alert_settings['volume_spike_threshold']
        )
    with col_s2:
        st.session_state.alert_settings['stakeholder_change_threshold'] = st.slider(
            "Stakeholder Change Threshold (% change)",
            min_value=0.1, max_value=5.0, step=0.1, format="%.1f%%",
            value=st.session_state.alert_settings['stakeholder_change_threshold']
        )
    
    st.success("Settings saved automatically.")

def main():
    st.markdown('<h1 class="main-header">🚀 NSE Pro Screener & Analytics</h1>', unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.title("🎯 Control Center")
        
        # Quick Stats
        if not st.session_state.stock_data.empty:
            total_stocks = len(st.session_state.stock_data)
            buy_signals = len(st.session_state.stock_data[
                st.session_state.stock_data.get('Enhanced_Score', 0) >= 70
            ])
            volume_alerts_count = len(st.session_state.stock_data[
                st.session_state.stock_data.get('Volume_Spike', False) == True
            ])
            stakeholder_alerts_count = len(st.session_state.stock_data[
                st.session_state.stock_data.get('Stakeholder_Alert', False) == True
            ])
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<h4>📊 Total Stocks: {total_stocks}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<h4>🎯 Strong Buy Signals: {buy_signals}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<h4>⚠️ Volume Spike Alerts: {volume_alerts_count}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<h4>👥 Stakeholder Change Alerts: {stakeholder_alerts_count}</h4>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        # Auto-update toggle
        auto_update = st.checkbox("🔄 Auto-Update (every 15 min)",
                                 value=st.session_state.auto_update_enabled,
                                 help="Automatically refresh data every 15 minutes. Note: Data fetching can be slow.")
        st.session_state.auto_update_enabled = auto_update

        st.info(f"Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'N/A'}")
        
        # Manual refresh button
        if st.button("Manual Refresh Data Now", use_container_width=True):
            st.session_state.trigger_update = True
            st.experimental_rerun() # Rerun to trigger the data fetch logic

    # Main tabs with enhanced features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Enhanced Screener", "🚨 Real-time Alerts", "📈 Technical Analysis",
        "💼 Smart Portfolio", "📰 Sentiment Analysis", "👥 Stakeholder Tracking", "⚙️ Settings"
    ])

    # Initialize screener
    screener = EnhancedNSEScreener()

    # Data fetching logic (triggered by button or auto-update)
    if st.session_state.trigger_update:
        with st.spinner("Fetching and analyzing latest stock data... This may take a few moments."):
            try:
                top_stocks = screener.get_nse_top_stocks(count=st.session_state.get('selected_stock_count', 50))
                # Run the async function
                st.session_state.stock_data = asyncio.run(screener.fetch_comprehensive_data(top_stocks))
                st.session_state.last_update = datetime.now()
                st.session_state.trigger_update = False # Reset the flag
                st.success("Data refreshed successfully!")
            except Exception as e:
                st.error(f"Failed to fetch data: {e}. Please try again later.")
                st.session_state.trigger_update = False


    with tab1:
        st.header("📊 Enhanced Stock Screener")

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            stock_count = st.selectbox("Number of Stocks to Analyze:", [50, 100, 200], index=0, key='selected_stock_count')
            st.session_state.selected_stock_count = stock_count # Save for refresh

            if st.button("Fetch & Analyze Data", type="primary", use_container_width=True):
                st.session_state.trigger_update = True
                st.experimental_rerun() # Rerun to trigger the data fetch logic

        with col2:
            min_score = st.slider("Min. Enhanced Score:", 0, 100, 60)
        with col3:
            selected_sector = st.selectbox("Filter by Sector:", 
                                         ['All'] + sorted(st.session_state.stock_data['Sector'].dropna().unique().tolist()))
        with col4:
            selected_risk = st.selectbox("Filter by Risk Level:", 
                                         ['All', 'Low Risk', 'Medium Risk', 'High Risk'])

        if not st.session_state.stock_data.empty:
            filtered_data = st.session_state.stock_data[st.session_state.stock_data['Enhanced_Score'] >= min_score]
            
            if selected_sector != 'All':
                filtered_data = filtered_data[filtered_data['Sector'] == selected_sector]
            if selected_risk != 'All':
                filtered_data = filtered_data[filtered_data['Risk_Level'] == selected_risk]
            
            st.subheader("Screener Results:")
            if not filtered_data.empty:
                # Add 'Status' column based on score for quick glance
                filtered_data['Recommendation_Status'] = filtered_data['Enhanced_Score'].apply(
                    lambda x: 'Strong Buy' if x >= 80 else ('Buy' if x >= 65 else ('Hold' if x >= 50 else 'Sell'))
                )
                
                # Format numerical columns for better display
                display_cols = [
                    'Symbol', 'Name', 'Current_Price', 'Change_Percent', 'Volume_Ratio',
                    'PE_Ratio', 'ROE', 'Debt_Equity', 'RSI', 'Technical_Signal',
                    'News_Sentiment', 'Sentiment_Label', 'Recommendation_Status',
                    'Enhanced_Score', 'Risk_Level', 'Sector', 'Industry'
                ]
                
                # Ensure all display columns exist, add missing with N/A
                for col in display_cols:
                    if col not in filtered_data.columns:
                        filtered_data[col] = 'N/A'

                st.dataframe(
                    filtered_data[display_cols].sort_values(by='Enhanced_Score', ascending=False).style.format({
                        'Current_Price': "{:,.2f}",
                        'Change_Percent': "{:+.2f}%",
                        'Volume_Ratio': "{:.2f}x",
                        'PE_Ratio': "{:,.2f}",
                        'ROE': "{:,.2f}%",
                        'Debt_Equity': "{:,.2f}",
                        'RSI': "{:,.2f}",
                        'News_Sentiment': "{:+.3f}",
                        'Enhanced_Score': "{}"
                    }).applymap(lambda x: 'background-color: #d4edda' if 'Buy' in x else ('background-color: #f8d7da' if 'Sell' in x else ''),
                                subset=['Recommendation_Status']),
                    use_container_width=True
                )
                
                st.markdown("---")
                st.subheader("Key Metrics Overview")
                
                # Dynamic metrics
                avg_pe = filtered_data['PE_Ratio'].replace([np.inf, -np.inf], np.nan).dropna().mean()
                avg_roe = filtered_data['ROE'].replace([np.inf, -np.inf], np.nan).dropna().mean()
                avg_volume_ratio = filtered_data['Volume_Ratio'].mean()
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("Avg. Enhanced Score", f"{filtered_data['Enhanced_Score'].mean():.1f}")
                with metrics_col2:
                    st.metric("Avg. PE Ratio", f"{avg_pe:.2f}" if not np.isnan(avg_pe) else "N/A")
                with metrics_col3:
                    st.metric("Avg. ROE", f"{avg_roe:.2f}%" if not np.isnan(avg_roe) else "N/A")
                with metrics_col4:
                    st.metric("Avg. Volume Ratio", f"{avg_volume_ratio:.2f}x")

                st.markdown("---")
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=filtered_data.to_csv(index=False).encode('utf-8'),
                    file_name="nse_screener_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            else:
                st.info("No stocks match the current filter criteria.")
        else:
            st.info("No data available. Click 'Fetch & Analyze Data' to load stocks.")

    with tab2:
        display_alerts()

    with tab3:
        display_technical_analysis()

    with tab4:
        display_smart_portfolio()

    with tab5:
        display_sentiment_analysis()

    with tab6:
        display_stakeholder_tracking()
        
    with tab7:
        display_settings()

if __name__ == "__main__":
    main()
