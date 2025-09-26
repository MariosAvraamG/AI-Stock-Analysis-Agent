import time
import requests
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from app.models.schemas import ToolResult
from app.core.config import settings

class AlphaVantageDataTool:
    """Alternative data source using Alpha Vantage API"""
    
    def __init__(self):
        self.name = "alpha_vantage_data"
        self.api_key = settings.alpha_vantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_ttl = 900  #15 minutes
        # Separate cache for historical data
        self.historical_cache = {}
        self.historical_cache_ttl = 1800  #30 minutes
        # Separate cache for fundamental data
        self.fundamental_cache = {}
        self.fundamental_cache_ttl = 3600  #1 hour (fundamental data changes even less frequently)
    
    def get_stock_data(self, ticker: str) -> ToolResult:
        """Fetch stock data from Alpha Vantage"""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"av_{ticker.upper()}"
            if cache_key in self.cache:
                cached_result, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    print(f"📦 Using cached Alpha Vantage data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching Alpha Vantage data for {ticker}")
            
            # Get daily prices
            daily_data = self._get_daily_prices(ticker)
            
            # Get company overview
            overview_data = self._get_company_overview(ticker)
            
            # Process the data
            result_data = self._process_alpha_vantage_data(ticker, daily_data, overview_data)
            
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache result
            self.cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _get_daily_prices(self, ticker: str) -> Dict[str, Any]:
        """Get daily price data from Alpha Vantage"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'apikey': self.api_key,
            'outputsize': 'compact'  #Last 100 days
        }
        
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError("Alpha Vantage API limit reached")
        
        return data
    
    def _get_company_overview(self, ticker: str) -> Dict[str, Any]:
        """Get company overview from Alpha Vantage"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
        except:
            return {}
    
    def _process_alpha_vantage_data(self, ticker: str, daily_data: Dict, overview_data: Dict) -> Dict[str, Any]:
        """Process Alpha Vantage data into our format"""
        
        time_series = daily_data.get('Time Series (Daily)', {})
        if not time_series:
            raise ValueError("No time series data found")
        
        # Get sorted dates
        dates = sorted(time_series.keys(), reverse=True)
        if not dates:
            raise ValueError("No price data found")
        
        # Current price (most recent close)
        current_data = time_series[dates[0]]
        current_price = float(current_data['4. close'])
        
        # Calculate performance
        performance = {}
        if len(dates) >= 2:
            prev_close = float(time_series[dates[1]]['4. close'])
            performance["1_day_change"] = ((current_price - prev_close) / prev_close) * 100
        
        if len(dates) >= 5:
            week_ago_close = float(time_series[dates[4]]['4. close'])
            performance["1_week_change"] = ((current_price - week_ago_close) / week_ago_close) * 100
        
        if len(dates) >= 20:
            month_ago_close = float(time_series[dates[19]]['4. close'])
            performance["1_month_change"] = ((current_price - month_ago_close) / month_ago_close) * 100
        
        # Volume
        current_volume = int(current_data['5. volume'])
        
        return {
            "ticker": ticker.upper(),
            "company_name": overview_data.get("Name", f"{ticker} Corporation"),
            "sector": overview_data.get("Sector", "Unknown"),
            "current_price": current_price,
            "currency": "USD",
            "market_cap": int(overview_data.get("MarketCapitalization", 0)) if overview_data.get("MarketCapitalization") else None,
            "pe_ratio": float(overview_data.get("PERatio", 0)) if overview_data.get("PERatio") != "None" else None,
            "dividend_yield": float(overview_data.get("DividendYield", 0)) if overview_data.get("DividendYield") != "None" else None,
            "52_week_high": float(overview_data.get("52WeekHigh", 0)) if overview_data.get("52WeekHigh") else None,
            "52_week_low": float(overview_data.get("52WeekLow", 0)) if overview_data.get("52WeekLow") else None,
            "performance": performance,
            "volume": {
                "current": current_volume,
                "average_1m": current_volume,  # Alpha Vantage doesn't provide avg volume
                "ratio": 1.0
            },
            "last_updated": datetime.now().isoformat(),
            "data_quality": "good",
            "info_available": len(overview_data) > 5,
            "data_source": "alpha_vantage"
        }

    def get_historical_data(self, ticker: str, period: str = "1y") -> ToolResult:
        """Fetch historical OHLCV data from Alpha Vantage"""
        start_time = time.time()
        
        try:
            # Check historical cache first
            cache_key = f"av_hist_{ticker.upper()}_{period}"
            if cache_key in self.historical_cache:
                cached_result, cached_time = self.historical_cache[cache_key]
                if time.time() - cached_time < self.historical_cache_ttl:
                    print(f"📦 Using cached Alpha Vantage historical data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching Alpha Vantage historical data for {ticker}")
            
            # Get daily prices with compact size (last 100 days)
            daily_data = self._get_daily_prices(ticker)
            
            # Process into DataFrame format
            result_data = self._process_historical_data(ticker, daily_data, period)
            
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache result
            self.historical_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            print(f"❌ Alpha Vantage historical data error for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _process_historical_data(self, ticker: str, daily_data: Dict, period: str) -> Dict[str, Any]:
        """Process Alpha Vantage data into DataFrame format for technical analysis"""
        
        time_series = daily_data.get('Time Series (Daily)', {})
        if not time_series:
            raise ValueError("No time series data found")
        
        # Convert to DataFrame format expected by technical analysis
        data_rows = []
        for date_str, values in time_series.items():
            try:
                data_rows.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })
            except (ValueError, KeyError) as e:
                print(f"⚠️ Skipping invalid data for {date_str}: {e}")
                continue
        
        if not data_rows:
            raise ValueError("No valid price data found")
        
        # Create DataFrame with Date as index (same format as yfinance)
        df = pd.DataFrame(data_rows)
        df = df.set_index('Date').sort_index()
        
        # Prepare result data
        result_data = {
            "ticker": ticker.upper(),
            "period": period,
            "historical_data": df,  # pandas DataFrame for technical analysis
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].strftime("%Y-%m-%d") if len(df) > 0 else None,
                "end": df.index[-1].strftime("%Y-%m-%d") if len(df) > 0 else None
            },
            "data_quality": "good" if len(df) > 50 else "limited",
            "columns": list(df.columns),
            "last_updated": datetime.now().isoformat(),
            "source": "alpha_vantage",
            "note": "Limited to last 100 days (compact size) to avoid rate limiting"
        }
        
        print(f"✅ Successfully processed {len(df)} days of Alpha Vantage historical data for {ticker}")
        return result_data
    
    def get_fundamental_data(self, ticker: str) -> ToolResult:
        """Fetch comprehensive fundamental data from Alpha Vantage"""
        start_time = time.time()
        
        try:
            # Check fundamental cache first
            cache_key = f"av_fund_{ticker.upper()}"
            if cache_key in self.fundamental_cache:
                cached_result, cached_time = self.fundamental_cache[cache_key]
                if time.time() - cached_time < self.fundamental_cache_ttl:
                    print(f"📦 Using cached Alpha Vantage fundamental data for {ticker}")
                    return cached_result
            
            print(f"🔍 Fetching Alpha Vantage fundamental data for {ticker}")
            
            # Get company overview (comprehensive fundamental data)
            overview_data = self._get_company_overview_comprehensive(ticker)
            
            # Process into standardized format
            result_data = self._process_alpha_vantage_fundamental_data(ticker, overview_data)
            
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache result
            self.fundamental_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            print(f"❌ Alpha Vantage fundamental data error for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _get_company_overview_comprehensive(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive company overview from Alpha Vantage API"""
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Symbol' not in data:
            raise ValueError(f"No fundamental data found for {ticker}")
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError("Alpha Vantage API limit reached")
        
        return data
    
    def _process_alpha_vantage_fundamental_data(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Alpha Vantage fundamental data into standardized format"""
        
        def safe_float(value):
            try:
                return float(value) if value and value != "None" and value != "-" else None
            except (ValueError, TypeError):
                return None
        
        def safe_int(value):
            try:
                return int(value) if value and value != "None" and value != "-" else None
            except (ValueError, TypeError):
                return None
        
        def safe_get(key, default=None):
            value = data.get(key, default)
            return value if value and value != "None" and value != "-" else default
        
        return {
            'ticker': ticker.upper(),
            'source': 'alpha_vantage',
            'last_updated': datetime.now().isoformat(),
            
            # Company Info
            'company_name': safe_get('Name'),
            'sector': safe_get('Sector'),
            'industry': safe_get('Industry'),
            'description': safe_get('Description'),
            'employees': safe_int(safe_get('FullTimeEmployees')),
            'exchange': safe_get('Exchange'),
            'currency': safe_get('Currency'),
            'country': safe_get('Country'),
            'address': safe_get('Address'),
            
            # Market Data
            'market_cap': safe_int(safe_get('MarketCapitalization')),
            'beta': safe_float(safe_get('Beta')),
            '52_week_high': safe_float(safe_get('52WeekHigh')),
            '52_week_low': safe_float(safe_get('52WeekLow')),
            
            # Valuation Ratios
            'pe_ratio': safe_float(safe_get('PERatio')),
            'peg_ratio': safe_float(safe_get('PEGRatio')),
            'pb_ratio': safe_float(safe_get('PriceToBookRatio')),
            'ps_ratio': safe_float(safe_get('PriceToSalesRatioTTM')),
            'ev_revenue': safe_float(safe_get('EVToRevenue')),
            'ev_ebitda': safe_float(safe_get('EVToEBITDA')),
            
            # Financial Performance
            'revenue_ttm': safe_int(safe_get('RevenueTTM')),
            'gross_profit_ttm': safe_int(safe_get('GrossProfitTTM')),
            'ebitda': safe_int(safe_get('EBITDA')),
            'net_income': safe_int(safe_get('NetIncomeTTM')),
            'eps': safe_float(safe_get('EPS')),
            'diluted_eps_ttm': safe_float(safe_get('DilutedEPSTTM')),
            
            # Growth Rates
            'revenue_growth_yoy': safe_float(safe_get('RevenueGrowthYoY')),
            'earnings_growth_yoy': safe_float(safe_get('EarningsGrowthYoY')),
            'quarterly_revenue_growth': safe_float(safe_get('QuarterlyRevenueGrowthYOY')),
            'quarterly_earnings_growth': safe_float(safe_get('QuarterlyEarningsGrowthYOY')),
            
            # Profitability Metrics
            'profit_margin': safe_float(safe_get('ProfitMargin')),
            'operating_margin': safe_float(safe_get('OperatingMarginTTM')),
            'gross_margin': safe_float(safe_get('GrossMarginTTM')),
            'roe': safe_float(safe_get('ReturnOnEquityTTM')),
            'roa': safe_float(safe_get('ReturnOnAssetsTTM')),
            
            # Financial Health
            'book_value': safe_float(safe_get('BookValue')),
            'debt_to_equity': safe_float(safe_get('DebtToEquityRatio')),
            'current_ratio': safe_float(safe_get('CurrentRatio')),
            'quick_ratio': safe_float(safe_get('QuickRatio')),
            
            # Dividend Info
            'dividend_per_share': safe_float(safe_get('DividendPerShare')),
            'dividend_yield': safe_float(safe_get('DividendYield')),
            'dividend_date': safe_get('DividendDate'),
            'ex_dividend_date': safe_get('ExDividendDate'),
            'payout_ratio': safe_float(safe_get('PayoutRatio')),
            
            # Share Info
            'shares_outstanding': safe_int(safe_get('SharesOutstanding')),
            'float_shares': safe_int(safe_get('SharesFloat')),
            'shares_short': safe_int(safe_get('SharesShort')),
            'short_ratio': safe_float(safe_get('ShortRatio')),
            'short_percent_outstanding': safe_float(safe_get('ShortPercentOutstanding')),
            'short_percent_float': safe_float(safe_get('ShortPercentFloat')),
            
            # Analyst Info
            'analyst_target_price': safe_float(safe_get('AnalystTargetPrice')),
            'analyst_rating_strong_buy': safe_int(safe_get('AnalystRatingStrongBuy')),
            'analyst_rating_buy': safe_int(safe_get('AnalystRatingBuy')),
            'analyst_rating_hold': safe_int(safe_get('AnalystRatingHold')),
            'analyst_rating_sell': safe_int(safe_get('AnalystRatingSell')),
            'analyst_rating_strong_sell': safe_int(safe_get('AnalystRatingStrongSell')),
            
            # Additional Alpha Vantage specific metrics
            'fiscal_year_end': safe_get('FiscalYearEnd'),
            'most_recent_quarter': safe_get('MostRecentQuarter'),
            'trailing_pe': safe_float(safe_get('TrailingPE')),
            'forward_pe': safe_float(safe_get('ForwardPE')),
        }

# Create instance
alpha_vantage_tool = AlphaVantageDataTool()