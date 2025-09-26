import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from app.models.schemas import ToolResult
from app.services.tools.multi_source_data import multi_source_tool

class TechnicalAnalysisTool:
    """
    Comprehensive technical analysis tool using pure pandas/numpy implementation
    """

    def __init__(self):
        self.name = "technical_analysis"
        self.cache = {}
        self.cache_ttl = 300 #5 minutes

    def _get_extended_historical_data(self, ticker: str, period: str="1y") -> pd.DataFrame:
        """Fetch extended historical data using multi-source approach"""
        try:
            print(f"🔍 Fetching historical data for technical analysis: {ticker}")
            
            # Use multi-source data tool for reliability
            result = multi_source_tool.get_historical_data(ticker, period)
            
            if not result.success:
                raise ValueError(f"Failed to fetch historical data: {result.error_message}")
            
            # Extract the DataFrame from the result
            hist = result.data.get('historical_data')
            if hist is None or hist.empty:
                raise ValueError(f"No historical data found for {ticker}")
            
            # Ensure we have enough data points for technical analysis
            if len(hist) < 50:
                print(f"⚠️ Limited data for {ticker}: {len(hist)} days")
            
            # Log data source and quality info
            source = result.data.get('primary_source', 'unknown')
            data_quality = result.data.get('data_quality', 'unknown')
            print(f"📊 Using {len(hist)} days of data from {source} (quality: {data_quality})")
            
            return hist
        
        except Exception as e:
            raise ValueError(f"Failed to fetch historical data for {ticker}: {str(e)}")
    
    def _calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=window, min_periods=window).mean()

    def _calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    def _calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(data, window)
        std = data.rolling(window=window).std()

        upper_band = sma + (num_std * std)
        middle_band = sma
        lower_band = sma - (num_std * std)

        return upper_band, middle_band, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()

        return k_percent, d_percent

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various moving averages"""
        try:
            close_prices = df['Close']

            #Simple Moving Averages
            sma_20 = self._calculate_sma(close_prices, 20)
            sma_50 = self._calculate_sma(close_prices, 50)
            sma_200 = self._calculate_sma(close_prices, 200)

            #Exponential Moving Averages
            ema_12 = self._calculate_ema(close_prices, 12)
            ema_26 = self._calculate_ema(close_prices, 26)
            ema_50 = self._calculate_ema(close_prices, 50)

            #Current value(most recent)
            current_price = close_prices.iloc[-1]

            ma_signals = {
                'current_price': current_price,
                'sma_20_current': sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50_current': sma_50.iloc[-1] if not pd.isna(sma_50.iloc[-1]) else None,
                'sma_200_current': sma_200.iloc[-1] if not pd.isna(sma_200.iloc[-1]) else None,
                'ema_12_current': ema_12.iloc[-1] if not pd.isna(ema_12.iloc[-1]) else None,
                'ema_26_current': ema_26.iloc[-1] if not pd.isna(ema_26.iloc[-1]) else None,
                'ema_50_current': ema_50.iloc[-1] if not pd.isna(ema_50.iloc[-1]) else None
            }

            #Generate signals
            signals = []
            if ma_signals['sma_20_current'] and ma_signals['sma_50_current']:
                if ma_signals['sma_20_current'] > ma_signals['sma_50_current']:
                    signals.append("Golden Cross (SMA20 > SMA50)")
                else:
                    signals.append("Death Cross (SMA20 < SMA50)")

            if ma_signals['sma_50_current'] and ma_signals['sma_200_current']:
                if ma_signals['sma_50_current'] > ma_signals['sma_200_current']:
                    signals.append("Long-term Bullish (SMA50 > SMA200)")
                else:
                    signals.append("Long-term Bearish (SMA50 < SMA200)")

            #Price position relative to moving averages
            ma_position = []
            if ma_signals['sma_20_current']:
                if current_price > ma_signals['sma_20_current']:
                    ma_position.append("Above SMA20")
                else:
                    ma_position.append("Below SMA20")

            if ma_signals['sma_50_current']:
                if current_price > ma_signals['sma_50_current']:
                    ma_position.append("Above SMA50")
                else:
                    ma_position.append("Below SMA50")

            ma_signals['signals'] = signals
            ma_signals['position'] = ma_position

            return ma_signals

        except Exception as e:
            print(f"⚠️ Error calculating moving averages: {e}")
            return {'error': str(e)}


    def _calculate_rsi_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI and generate signals"""
        try:
            close_prices = df['Close']
            rsi = self._calculate_rsi(close_prices, 14)

            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

            #Generate RSI signals
            signal = "NEUTRAL"
            if current_rsi:
                if current_rsi > 70:
                    signal = "OVERBOUGHT"
                elif current_rsi < 30:
                    signal = "OVERSOLD"

            return {
                'current_rsi': current_rsi,
                'signal': signal,
                'interpretation': self._interpret_rsi(current_rsi)
            }

        except Exception as e:
            print(f"⚠️ Error calculating RSI: {e}")
            return {'error': str(e)}

    def _interpret_rsi(self, rsi: Optional[float]) -> str:
        """Interpret RSI value"""
        if not rsi:
            return "RSI data unavailable"

        if rsi > 70:
            return "Strong overbought condition - potential sell signal"
        elif rsi > 60:
            return "Moderately overbought - caution advised"
        elif rsi < 30:
            return "Strong oversold condition - potential buy signal"
        elif rsi < 40:
            return "Moderately oversold - potential opportunity"
        else:
            return "RSI in neutral range"

    def _calculate_macd_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD and generate signals"""
        try:
            close_prices = df['Close']
            macd, macd_signal, macd_histogram = self._calculate_macd(close_prices)

            current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else None
            current_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else None
            current_histogram = macd_histogram.iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else None
        
            #Generate MACD signals
            signal = "NEUTRAL"
            if current_macd and current_signal:
                if current_macd > current_signal:
                    signal = "BULLISH"
                else:
                    signal = "BEARISH"

            #Check for crossover
            crossover = None
            if len(macd) > 1:
                prev_macd = macd.iloc[-2] if not pd.isna(macd.iloc[-2]) else None
                prev_signal = macd_signal.iloc[-2] if not pd.isna(macd_signal.iloc[-2]) else None

                if prev_macd and prev_signal and current_macd and current_signal:
                    if prev_macd <= prev_signal and current_macd > current_signal:
                        crossover = "BULLISH CROSSOVER"
                    elif prev_macd >= prev_signal and current_macd < current_signal:
                        crossover = "BEARISH CROSSOVER"

                return {
                    'macd': current_macd,
                    'signal_line': current_signal,
                    'histogram': current_histogram,
                    'signal': signal,
                    'crossover': crossover
                }

        except Exception as e:
            print(f"⚠️ Error calculating MACD: {e}")
            return {'error': str(e)}

    def _calculate_bollinger_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands and generate signals"""
        try:
            close_prices = df['Close']
            upper, middle, lower = self._calculate_bollinger_bands(close_prices)
            
            current_price = close_prices.iloc[-1]
            current_upper = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else None
            current_middle = middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else None
            current_lower = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else None

            #Generate signals
            signal = "NEUTRAL"
            position = "MIDDLE"

            if current_upper and current_lower:
                if current_price > current_upper:
                    signal = "OVERBOUGHT"
                    position = "ABOVE UPPER"
                elif current_price < current_lower:
                    signal = "OVERSOLD"
                    position = "BELOW LOWER"
            
            #Calculate Band Width
            band_width = ((current_upper - current_lower) / current_middle) * 100 if current_middle else None

            return {
                'upper_band': current_upper,
                'middle_band': current_middle,
                'lower_band': current_lower,
                'signal': signal,
                'position': position,
                'band_width_pct': band_width
            }
        
        except Exception as e:
            print(f"⚠️ Error calculating Bollinger Bands: {e}")
            return {'error': str(e)}
        
    def _calculate_stochastic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator and generate signals"""
        try:
            high_prices = df['High']
            low_prices = df['Low']
            close_prices = df['Close']
            k_percent, d_percent = self._calculate_stochastic(high_prices, low_prices, close_prices)

            current_k = k_percent.iloc[-1] if not pd.isna(k_percent.iloc[-1]) else None
            current_d = d_percent.iloc[-1] if not pd.isna(d_percent.iloc[-1]) else None

            #Generate signals
            signal = "NEUTRAL"
            if current_k and current_d:
                if current_k > 80 and current_d > 80:
                    signal = "OVERBOUGHT"
                elif current_k < 20 and current_d < 20:
                    signal = "OVERSOLD"

            return {
                'stock_k': current_k,
                'stock_d': current_d,
                'signal': signal,
            }

        except Exception as e:
            print(f"⚠️ Error calculating Stochastic Oscillator: {e}")
            return {'error': str(e)}

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Support and Resistance levels using pivot points"""
        try:
            #Use recent 60 days for support/resistance calculation
            recent_data = df.tail(60) if len(df) > 60 else df

            highs = recent_data['High']
            lows = recent_data['Low']
            closes = recent_data['Close']

            #Find pivot points using local maxima and minima
            resistance_levels = []
            support_levels = []

            #Convert to numpy arrays for easier processing
            high_values = highs.values
            low_values = lows.values

            #Simple pivot point calculation
            for i in range(2, len(high_values) - 2):
                #Resistance: high point higher than surrounding points
                if (high_values[i] > high_values[i-1] and high_values[i] > high_values[i+1]
                    and high_values[i] > high_values[i-2] and high_values[i] > high_values[i+2]):
                    resistance_levels.append(high_values[i])
                
                #Support: low point lower than surrounding points
                if (low_values[i] < low_values[i-1] and low_values[i] < low_values[i+1]
                    and low_values[i] < low_values[i-2] and low_values[i] < low_values[i+2]):
                    support_levels.append(low_values[i])

            #Get current price for comparison
            current_price = closes.iloc[-1]

            #Find nearest support and resistance
            resistance_levels = sorted(set(resistance_levels), reverse=True)
            support_levels = sorted(set(support_levels), reverse=True)

            nearest_resistance = None
            nearest_support = None
            
            for level in resistance_levels:
                if level > current_price:
                    nearest_resistance = level
                    break
            
            for level in support_levels:
                if level < current_price:
                    nearest_support = level
                    break

            return {
                'current_price': current_price,
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'resistance_levels': resistance_levels[:5],  # Top 5
                'support_levels': support_levels[:5]  # Top 5
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating support/resistance: {e}")
            return {'error': str(e)}


    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        try:
            close_prices = df['Close']
            volume = df['Volume']

            # Volume Moving Average
            volume_sma_20 = self._calculate_sma(volume, 20)

            #Simple On-Balance Volume calculation
            obv = []
            obv_value = 0

            for i in range(len(df)):
                if i == 0:
                    obv.append(volume.iloc[i])
                    obv_value = volume.iloc[i]
                else:
                    if close_prices.iloc[i] > close_prices.iloc[i-1]:
                        obv_value += volume.iloc[i]
                    elif close_prices.iloc[i] < close_prices.iloc[i-1]:
                        obv_value -= volume.iloc[i]
                    #Note: If price is unchanged, OBV remains unchanged
                    obv.append(obv_value)

            current_volume = volume.iloc[-1]
            avg_volume = volume_sma_20.iloc[-1] if not pd.isna(volume_sma_20.iloc[-1]) else None
            current_obv = obv[-1] if obv else None

            #Volume Signal
            volume_signal = "NORMAL"
            volume_ratio = None
            if avg_volume and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                if volume_ratio > 2.0:
                    volume_signal = "HIGH VOLUME"
                elif volume_ratio < 0.5:
                    volume_signal = "LOW VOLUME"
            
            return {
                'current_volume': int(current_volume),
                'avg_volume_20': int(avg_volume) if avg_volume else None,
                'volume_ratio': round(volume_ratio, 2) if volume_ratio else None,
                'volume_signal': volume_signal,
                'obv': current_obv
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating volume indicators: {e}")
            return {'error': str(e)}

    def _generate_overall_signal(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall trading signal based on all indicators"""
        signals = []
        bullish_count = 0
        bearish_count = 0

        #Moving Average signals
        ma = indicators.get('moving_averages', {})
        if 'signals' in ma:
            for signal in ma['signals']:
                if 'Golden Cross' in signal or 'Long-term Bullish' in signal:
                    bullish_count += 1
                    signals.append(f"MA: {signal}")
                elif 'Death Cross' in signal or 'Long-term Bearish' in signal:
                    bearish_count += 1
                    signals.append(f"MA: {signal}")
        
        #Price position relative to moving averages
        if 'position' in ma:
            above_count = sum(1 for pos in ma['position'] if 'Above' in pos)
            below_count = len(ma['position']) - above_count
            if above_count > below_count:
                bullish_count += 0.5
                signals.append("Price above key moving averages")
            elif below_count > above_count:
                bearish_count += 0.5
                signals.append("Price below key moving averages")
        
        #RSI signals
        rsi = indicators.get('rsi', {})
        if rsi.get('signal') == "OVERSOLD":
            bullish_count += 1
            signals.append("RSI: Oversold condition")
        elif rsi.get('signal') == "OVERBOUGHT":
            bearish_count += 1
            signals.append("RSI: Overbought condition")

        #MACD signals
        macd = indicators.get('macd', {})
        if macd.get('signal') == "BULLISH":
            bullish_count += 1
            signals.append("MACD: Bullish trend")
        elif macd.get('signal') == "BEARISH":
            bearish_count += 1
            signals.append("MACD: Bearish trend")
        
        #Crossovers are stronger signals
        if macd.get('crossover') == "BULLISH CROSSOVER":
            bullish_count += 1.5
            signals.append("MACD: Bullish crossover")
        elif macd.get('crossover') == "BEARISH CROSSOVER":
            bearish_count += 1.5
            signals.append("MACD: Bearish crossover")
        
        #Bollinger Bands signals
        bollinger = indicators.get('bollinger', {})
        if bollinger.get('signal') == "OVERSOLD":
            bullish_count += 1
            signals.append("Bollinger: Oversold condition")
        elif bollinger.get('signal') == "OVERBOUGHT":
            bearish_count += 1
            signals.append("Bollinger: Overbought condition")
        
        #Stochastic signals
        stochastic = indicators.get('stochastic', {})
        if stochastic.get('signal') == "OVERSOLD":
            bullish_count += 1
            signals.append("Stochastic: Oversold condition")
        elif stochastic.get('signal') == "OVERBOUGHT":
            bearish_count += 1
            signals.append("Stochastic: Overbought condition")
        
        #Volume confirmation
        volume = indicators.get('volume_indicators', {})
        if volume.get('volume_signal') == "HIGH VOLUME":
            signals.append("High volume activity")
        
        #Generate overall signal
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            overall_signal = "HOLD"
            confidence = 0.3
        else:
            bullish_ratio = bullish_count / total_signals
            if bullish_ratio >= 0.65:
                overall_signal = "BUY"
                confidence = min(bullish_ratio, 0.95)
            elif bullish_ratio <= 0.35:
                overall_signal = "SELL"
                confidence = min(1 - bullish_ratio, 0.95)
            else:
                overall_signal = "HOLD"
                confidence = 0.5
        
        return {
            'signal': overall_signal,
            'confidence': round(confidence, 2),
            'bullish_indicators': bullish_count,
            'bearish_indicators': bearish_count,
            'supporting_signals': signals
        }

    def _analyze_trend(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Analyze trend for a specific period"""
        if len(df) < period:
            return {'trend': 'INSUFFICIENT DATA', 'strength': 0}

        try:
            recent_data = df.tail(period)
            start_price = recent_data['Close'].iloc[0]
            end_price = recent_data['Close'].iloc[-1]

            price_change = ((end_price - start_price) / start_price) * 100

            #Calculate trend strength using linear regression slope
            x = np.arange(len(recent_data))
            y = recent_data['Close'].values
            slope = np.polyfit(x, y, 1)[0]

            #Normalize slope to get strength
            avg_price = np.mean(y)
            normalized_slope = (slope / avg_price) * 100 * len(recent_data)

            #Determine trend
            if price_change > 10:
                trend = "STRONG UPTREND"
                strength = min(abs(normalized_slope)/20, 1.0)

            elif price_change > 3:
                trend = "UPTREND"
                strength = min(abs(normalized_slope)/15, 1.0)
            
            elif price_change < -10:
                trend = "STRONG DOWNTREND"
                strength = min(abs(normalized_slope)/20, 1.0)
            
            elif price_change < -3:
                trend = "DOWNTREND"
                strength = min(abs(normalized_slope)/15, 1.0)

            else:
                trend = "SIDEWAYS"
                strength = max(0.2, min(abs(normalized_slope)/30, 0.6))

            return {
                'trend': trend,
                'strength': round(strength, 2),
                'price_change_pct': round(price_change, 2),
                'period_days': period
            }
            
        except Exception as e:
            return {'trend': 'ERROR', 'error': str(e)}

    def analyze_stock(self, ticker: str, period: str = "1y") -> ToolResult:
        """Perform comprehensive technical analysis"""
        start_time = time.time()

        try:
            #Check cache first
            cache_key = f"ta_{ticker.upper()}_{period}"
            if cache_key in self.cache:
                cached_result, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    print(f"📦 Using cached technical analysis for {ticker}")
                    return cached_result
            
            print(f"🔍 Performing technical analysis for {ticker}")

            #Get historical data
            df = self._get_extended_historical_data(ticker, period)

            #Calculate all technical indicators
            indicators = {
                'moving_averages': self._calculate_moving_averages(df),
                'rsi': self._calculate_rsi_analysis(df),
                'macd': self._calculate_macd_analysis(df),
                'bollinger': self._calculate_bollinger_analysis(df),
                'stochastic': self._calculate_stochastic_analysis(df),
                'support_resistance': self._calculate_support_resistance(df),
                'volume_indicators': self._calculate_volume_indicators(df)
            }

            #Generate overall signal
            overall_analysis = self._generate_overall_signal(indicators)

            #Prepare result data
            result_data = {
                'ticker': ticker.upper(),
                'analysis_period': period,
                'data_points': len(df),
                'last_update': datetime.now().isoformat(),
                'indicators': indicators,
                'overall_analysis': overall_analysis,
                'trend_analysis': {
                    'short_term': self._analyze_trend(df, 20),
                    'medium_term': self._analyze_trend(df, 50),
                    'long_term': self._analyze_trend(df, 200)
                }
            }

            #Create ToolResult object
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
            
            #Cache result
            self.cache[cache_key] = (result, time.time())

            print(f"✅ Technical analysis completed for {ticker}")
            return result
        
        except Exception as e:
            print(f"❌ Error in technical analysis for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )

#Global instance
technical_analysis_tool = TechnicalAnalysisTool()