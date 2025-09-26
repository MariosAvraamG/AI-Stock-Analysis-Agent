import yfinance as yf
import pandas as pd
import time
import random
from typing import Dict, Any
from datetime import datetime
from app.models.schemas import ToolResult

class MarketDataTool:
    """Market data fetcher with rate limiting and rety logic"""

    def __init__(self):
        self.name = "market_data"
        self.cache = {}
        self.cache_ttl = 300 #5 minutes
        self.last_request_time = 0
        self.min_request_interval = 2 #2 seconds between requests
        # Separate cache for historical data
        self.historical_cache = {}
        self.historical_cache_ttl = 1800 #30 minutes (historical data changes less frequently)
        # Separate cache for fundamental data
        self.fundamental_cache = {}
        self.fundamental_cache_ttl = 3600 #1 hour (fundamental data changes even less frequently)

    def _wait_for_rate_limit(self):
        """Ensure we don't hit rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _calculate_performance(self, hist_5d, hist_1m, current_price) -> Dict[str, float]:
        """Calculate performance metrics"""
        performance = {}
        
        try:
            if not hist_5d.empty and len(hist_5d) >= 2:
                prev_close = float(hist_5d['Close'].iloc[-2])
                performance["1_day_change"] = ((current_price - prev_close) / prev_close) * 100
            
            if not hist_5d.empty and len(hist_5d) >= 5:
                week_start = float(hist_5d['Close'].iloc[0])
                performance["1_week_change"] = ((current_price - week_start) / week_start) * 100
            
            if not hist_1m.empty and len(hist_1m) >= 20:
                month_start = float(hist_1m['Close'].iloc[0])
                performance["1_month_change"] = ((current_price - month_start) / month_start) * 100
        
        except Exception as e:
            print(f"⚠️ Error calculating performance: {e}")
        
        return performance
    
    def _analyze_volume(self, recent_hist, hist_1m) -> Dict[str, Any]:
        """Analyze volume data"""
        try:
            current_volume = int(recent_hist['Volume'].iloc[-1])
            avg_volume = int(hist_1m['Volume'].mean()) if not hist_1m.empty else current_volume
            
            return {
                "current": current_volume,
                "average_1m": avg_volume,
                "ratio": current_volume / avg_volume if avg_volume > 0 else 1.0
            }
        except Exception as e:
            print(f"⚠️ Error analyzing volume: {e}")
            return {
                "current": 0,
                "average_1m": 0,
                "ratio": 1.0
            }

    def _fetch_with_retry(self, ticker: str, max_retries: int = 3) -> Dict[str, Any]:
        """Fetch stock data with retry logic"""

        for attempt in range(max_retries):
            try:
                print(f"📡 Attempt {attempt + 1}/{max_retries} for {ticker}")

                #Create ticker object
                stock = yf.Ticker(ticker)

                #Get basic info(this often fails with rate limiting)
                info = {}
                try:
                    info = stock.info
                    print(f"✅ Got company info for {ticker}")
                except Exception as e:
                    print(f"⚠️ Could not fetch company info: {str(e)[:100]}...")
                    # Continue without company info

                # Get price history (more reliable than info)
                print(f"📈 Fetching price history for {ticker}")
                hist_1m = stock.history(period='1mo')
                hist_5d = hist_1m.tail(5)

                if hist_1m.empty:
                    raise ValueError(f"No price data available for {ticker}")

                # Use the most recent data available
                recent_hist = hist_5d if not hist_5d.empty else hist_1m
                current_price = float(recent_hist['Close'].iloc[-1])

                print(f"💰 Current price for {ticker}: ${current_price:.2f}")

                # Calculate performance metrics
                performance = self._calculate_performance(hist_5d, hist_1m, current_price)
                
                # Volume analysis
                volume_data = self._analyze_volume(recent_hist, hist_1m)
                
                # Prepare result data
                result_data = {
                    "ticker": ticker,
                    "company_name": info.get("longName", f"{ticker} Corporation"),
                    "sector": info.get("sector", "Unknown"),
                    "current_price": current_price,
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "performance": performance,
                    "volume": volume_data,
                    "last_updated": datetime.now().isoformat(),
                    "data_quality": "good" if len(hist_1m) > 15 else "limited",
                    "info_available": len(info) > 5  # Whether we got company info
                }
                
                print(f"✅ Successfully fetched data for {ticker}")
                return result_data
            
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Wait with exponential backoff + jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts: {str(e)}")
    


    def get_stock_data(self, ticker: str) -> ToolResult:
        """Fetch basic stock market data"""
        start_time = time.time()

        try:
            #check cache first
            cache_key = f"{ticker.upper()}"
            if cache_key in self.cache:
                cached_result, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    print(f"📦 Using cached data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching fresh data for {ticker}")

            #Wait to avoid rate limiting
            self._wait_for_rate_limit()

            #Fetch data with retries
            stock_data = self._fetch_with_retry(ticker.upper())

            # Create successful result
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=stock_data,
                execution_time_seconds=time.time() - start_time
            )

            #Cache successful result
            self.cache[cache_key] = (result, time.time())

            return result

        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def get_historical_data(self, ticker: str, period: str = "1y") -> ToolResult:
        """Fetch historical OHLCV data for technical analysis"""
        start_time = time.time()
        
        try:
            # Check historical cache first
            cache_key = f"hist_{ticker.upper()}_{period}"
            if cache_key in self.historical_cache:
                cached_result, cached_time = self.historical_cache[cache_key]
                if time.time() - cached_time < self.historical_cache_ttl:
                    print(f"📦 Using cached historical data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching historical data for {ticker} (period: {period})")
            
            # Wait to avoid rate limiting
            self._wait_for_rate_limit()
            
            # Fetch historical data with retries
            historical_data = self._fetch_historical_with_retry(ticker.upper(), period)
            
            # Create successful result
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=historical_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache successful result
            self.historical_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            print(f"❌ Error fetching historical data for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _fetch_historical_with_retry(self, ticker: str, Period: str, max_retries: int = 3) -> Dict[str, Any]:
        """Fetch historical OHLCV data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                print(f"📡 Historical data attempt {attempt + 1}/{max_retries} for {ticker}")
                
                # Create ticker object
                stock = yf.Ticker(ticker)
                
                # Get historical data
                hist = stock.history(period=Period)
                
                if hist.empty:
                    raise ValueError(f"No historical data available for {ticker}")
                
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in hist.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                #Prepare result data optimized for technical analysis
                result_data = {
                    "ticker": ticker,
                    "period": Period,
                    "historical_data": hist,  #Full pandas DataFrame for technical analysis
                    "data_points": len(hist),
                    "date_range": {
                        "start": hist.index[0].strftime("%Y-%m-%d") if len(hist) > 0 else None,
                        "end": hist.index[-1].strftime("%Y-%m-%d") if len(hist) > 0 else None
                    },
                    "data_quality": "good" if len(hist) > 50 else "limited",
                    "columns": list(hist.columns),
                    "last_updated": datetime.now().isoformat(),
                    "source": "yfinance"
                }
                
                print(f"✅ Successfully fetched {len(hist)} days of historical data for {ticker}")
                return result_data
                
            except Exception as e:
                print(f"❌ Historical data attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Wait with exponential backoff + jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise ValueError(f"Failed to fetch historical data for {ticker} after {max_retries} attempts: {str(e)}")
    
    def get_fundamental_data(self, ticker: str) -> ToolResult:
        """Fetch comprehensive fundamental data for analysis"""
        start_time = time.time()
        
        try:
            # Check fundamental cache first
            cache_key = f"fund_{ticker.upper()}"
            if cache_key in self.fundamental_cache:
                cached_result, cached_time = self.fundamental_cache[cache_key]
                if time.time() - cached_time < self.fundamental_cache_ttl:
                    print(f"📦 Using cached fundamental data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching fundamental data for {ticker}")
            
            # Wait to avoid rate limiting
            self._wait_for_rate_limit()
            
            # Fetch fundamental data with retries
            fundamental_data = self._fetch_fundamental_with_retry(ticker.upper())
            
            # Create successful result
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=fundamental_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache successful result
            self.fundamental_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            print(f"❌ Error fetching fundamental data for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _fetch_fundamental_with_retry(self, ticker: str, max_retries: int = 3) -> Dict[str, Any]:
        """Fetch comprehensive fundamental data with retry logic"""
        
        for attempt in range(max_retries):
            try:
                print(f"📡 Fundamental data attempt {attempt + 1}/{max_retries} for {ticker}")
                
                # Create ticker object
                stock = yf.Ticker(ticker)
                
                # Get comprehensive info (this contains fundamental data)
                info = stock.info
                
                if not info:
                    raise ValueError(f"No fundamental data available for {ticker}")
                
                # Process and standardize the fundamental data
                result_data = self._process_yahoo_fundamental_data(ticker, info)
                
                print(f"✅ Successfully fetched fundamental data for {ticker}")
                return result_data
                
            except Exception as e:
                print(f"❌ Fundamental data attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Wait with exponential backoff + jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise ValueError(f"Failed to fetch fundamental data for {ticker} after {max_retries} attempts: {str(e)}")
    
    def _process_yahoo_fundamental_data(self, ticker: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Process Yahoo Finance fundamental data into standardized format"""
        
        def safe_get(key, default=None):
            value = info.get(key, default)
            return value if (value is not None) and (value != 'N/A') and (str(value).lower() != 'none') else default
        
        def safe_float(value):
            try:
                return float(value) if (value is not None) and (value != 'N/A') and (str(value).lower() != 'none') else None
            except (ValueError, TypeError):
                return None
        
        def safe_int(value):
            try:
                return int(value) if (value is not None) and (value != 'N/A') and (str(value).lower() != 'none')else None
            except (ValueError, TypeError):
                return None
        
        return {
            'ticker': ticker.upper(),
            'source': 'yfinance',
            'last_updated': datetime.now().isoformat(),
            
            # Company Info
            'company_name': safe_get('longName') or safe_get('shortName'),
            'sector': safe_get('sector'),
            'industry': safe_get('industry'),
            'description': safe_get('longBusinessSummary'),
            'employees': safe_int(safe_get('fullTimeEmployees')),
            'website': safe_get('website'),
            'city': safe_get('city'),
            'state': safe_get('state'),
            'country': safe_get('country'),
            
            # Market Data
            'market_cap': safe_int(safe_get('marketCap')),
            'current_price': safe_float(safe_get('currentPrice') or safe_get('regularMarketPrice')),
            'beta': safe_float(safe_get('beta')),
            '52_week_high': safe_float(safe_get('fiftyTwoWeekHigh')),
            '52_week_low': safe_float(safe_get('fiftyTwoWeekLow')),
            
            # Valuation Ratios
            'pe_ratio': safe_float(safe_get('trailingPE') or safe_get('forwardPE')),
            'peg_ratio': safe_float(safe_get('trailingPegRatio')),
            'pb_ratio': safe_float(safe_get('priceToBook')),
            'ps_ratio': safe_float(safe_get('priceToSalesTrailing12Months')),
            'enterprise_value': safe_int(safe_get('enterpriseValue')),
            'ev_revenue': safe_float(safe_get('enterpriseToRevenue')),
            'ev_ebitda': safe_float(safe_get('enterpriseToEbitda')),
            
            # Financial Performance
            'revenue_ttm': safe_int(safe_get('totalRevenue')),
            'gross_profit': safe_int(safe_get('grossProfits')),
            'ebitda': safe_int(safe_get('ebitda')),
            'net_income': safe_int(safe_get('netIncomeToCommon')),
            'eps': safe_float(safe_get('trailingEps')),
            'forward_eps': safe_float(safe_get('forwardEps')),
            
            # Growth Rates
            'revenue_growth': safe_float(safe_get('revenueGrowth')),
            'earnings_growth': safe_float(safe_get('earningsGrowth')),
            'earnings_quarterly_growth': safe_float(safe_get('earningsQuarterlyGrowth')),
            
            # Profitability Metrics
            'profit_margin': safe_float(safe_get('profitMargins')),
            'operating_margin': safe_float(safe_get('operatingMargins')),
            'gross_margin': safe_float(safe_get('grossMargins')),
            'roe': safe_float(safe_get('returnOnEquity')),
            'roa': safe_float(safe_get('returnOnAssets')),
            
            # Financial Health
            'book_value': safe_float(safe_get('bookValue')),
            'debt_to_equity': safe_float(safe_get('debtToEquity')),
            'current_ratio': safe_float(safe_get('currentRatio')),
            'quick_ratio': safe_float(safe_get('quickRatio')),
            'total_cash': safe_int(safe_get('totalCash')),
            'total_debt': safe_int(safe_get('totalDebt')),
            'free_cash_flow': safe_int(safe_get('freeCashflow')),
            'operating_cash_flow': safe_int(safe_get('operatingCashflow')),
            
            # Dividend Info
            'dividend_rate': safe_float(safe_get('dividendRate')),
            'dividend_yield': safe_float(safe_get('dividendYield')),
            'payout_ratio': safe_float(safe_get('payoutRatio')),
            'five_year_avg_dividend_yield': safe_float(safe_get('fiveYearAvgDividendYield')),
            'dividend_date': safe_get('dividendDate'),
            'ex_dividend_date': safe_get('exDividendDate'),
            
            # Share Info
            'shares_outstanding': safe_int(safe_get('sharesOutstanding')),
            'float_shares': safe_int(safe_get('floatShares')),
            'shares_short': safe_int(safe_get('sharesShort')),
            'short_ratio': safe_float(safe_get('shortRatio')),
            'short_percent_of_float': safe_float(safe_get('shortPercentOfFloat')),
            'held_percent_insiders': safe_float(safe_get('heldPercentInsiders')),
            'held_percent_institutions': safe_float(safe_get('heldPercentInstitutions')),
            
            # Analyst Info
            'target_high_price': safe_float(safe_get('targetHighPrice')),
            'target_low_price': safe_float(safe_get('targetLowPrice')),
            'target_mean_price': safe_float(safe_get('targetMeanPrice')),
            'target_median_price': safe_float(safe_get('targetMedianPrice')),
            'recommendation_mean': safe_float(safe_get('recommendationMean')),
            'recommendation_key': safe_get('recommendationKey'),
            'number_of_analyst_opinions': safe_int(safe_get('numberOfAnalystOpinions')),
            
            # Additional metrics
            'trailing_annual_dividend_rate': safe_float(safe_get('trailingAnnualDividendRate')),
            'trailing_annual_dividend_yield': safe_float(safe_get('trailingAnnualDividendYield')),
            'price_to_sales_trailing_12_months': safe_float(safe_get('priceToSalesTrailing12Months')),
            'forward_pe': safe_float(safe_get('forwardPE')),
            'trailing_pe': safe_float(safe_get('trailingPE')),
        }

#Global instance for testing
market_data_tool = MarketDataTool()