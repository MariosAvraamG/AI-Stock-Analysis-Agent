import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

#TensorFlow/Keras imports for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow import failed: {e}")

from app.models.schemas import ToolResult
from app.services.tools.multi_source_data import multi_source_tool

class MLPredictionTool:
    """
    Technical-Only ML Prediction Tool using ensemble of proven models:
    - Random Forest (Classification & Regression)
    - Gradient Boosting
    - Support Vector Machines
    - LSTM Neural Networks (if TensorFlow available)
    - Linear Models

    FOCUSED ON TECHNICAL ANALYSIS:
    - No fundamental features due to difficult extraction of matching historical fundamental data for training
    - No sentiment features due to difficulty in extracting matching historical sentiment data for training
    - Pure price/volume/technical indicator based predictions
    - Proper temporal separation with adaptive window sizes
    """

    def __init__(self):
        self.name = "ml_prediction"
        self.cache = {}
        self.cache_ttl = 3600 #1 hour cache

        #Prediction timeframes with different model strategies and adaptive windows
        self.timeframes = {
            'short_term': {
                'days': 5,
                'window_size': 20,  #20 days of features for 5-day prediction (4x)
                'weight': 0.4,
                'models': ['random_forest', 'svm', 'lstm'],
            },

            'medium_term': {
                'days': 20,
                'window_size': 60,  #60 days of features for 20-day prediction (3x)
                'weight': 0.35,
                'models': ['random_forest', 'gradient_boosting', 'lstm'],
            },

            'long_term': {
                'days': 60,
                'window_size': 120,  #120 days of features for 60-day prediction (2x)
                'weight': 0.25,
                'models': ['random_forest', 'gradient_boosting', 'linear_regression'],
            }
        }

    def _error_result(self, message: str, start_time: float) -> ToolResult:
        """Create error result"""
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={},
            error_message=message,
            execution_time_seconds=time.time() - start_time
        )

    def analyze_stock(self, ticker: str) -> ToolResult:
        """Main method for technical-only ML stock prediction analysis"""
        start_time = time.time()

        try:
            #Check cache
            cache_key = f"{ticker}_ml_prediction"
            if cache_key in self.cache:
                cache_data, cache_time = self.cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    print(f"📊 Using cached ML prediction for {ticker}")
                    return ToolResult(
                        tool_name=self.name,
                        success=True,
                        data=cache_data,
                        execution_time_seconds=time.time() - start_time
                    )

            print(f"🤖 Starting technical-only ML prediction analysis for {ticker}...")  

            #Step 1: Get historical data for training (need more data for long-term window)
            historical_data = multi_source_tool.get_historical_data(ticker, period='2y')
            if not historical_data.success:
                return self._error_result("Failed to gather historical data", start_time)
            
            #Step 2: Prepare timeframe-specific training datasets (technical features only)
            timeframe_datasets = self._prepare_technical_training_dataset(historical_data.data, ticker)
            if not timeframe_datasets:
                return self._error_result("Failed to prepare technical training datasets", start_time)
            
            #Step 3: Train timeframe-specific models
            timeframe_models = self._train_models(timeframe_datasets, ticker)

            #Step 4: Generate predictions for each timeframe using trained models
            predictions = {}
            for timeframe, config in self.timeframes.items():
                if timeframe in timeframe_models:
                    #Prepare current data for prediction
                    current_data = {
                        'historical_data': historical_data.data.get('historical_data')
                    }
                    
                    prediction = self._generate_timeframe_prediction(
                        timeframe, current_data, timeframe_models[timeframe]
                    )
                    predictions[timeframe] = prediction
                else:
                    #Fallback if timeframe model training failed
                    predictions[timeframe] = self._default_prediction(timeframe)

            #Step 5: Generate ensemble predictions
            ensemble_result = self._generate_ensemble_prediction(predictions)

            #Step 6: Calculate confidence and risk
            confidence_analysis = self._calculate_prediction_confidence(predictions)
            risk_assessment = self._assess_prediction_risk(predictions)

            #Step 7: Generate trading signals
            trading_signals = self._generate_ml_trading_signals(ensemble_result, confidence_analysis)

            #Compile results
            result_data = {
                'ticker': ticker,
                'analysis_timestamp': datetime.now().isoformat(),
                'approach': 'technical_only',
                'predictions': predictions,
                'ensemble_prediction': ensemble_result,
                'confidence_analysis': confidence_analysis,
                'risk_assessment': risk_assessment,
                'trading_signals': trading_signals,
                'model_performance': self._get_model_performance(),
                'feature_importance': self._get_feature_importance_summary(),
                'data_quality': self._assess_data_quality(),
                'model_metadata': {
                    'timeframe_models': {tf: list(tm.get('models', {}).keys()) for tf, tm in timeframe_models.items()},
                    'tensorflow_available': TENSORFLOW_AVAILABLE,
                    'training_samples': {tf: tm.get('training_samples', 0) for tf, tm in timeframe_models.items()},
                    'training_approach': 'technical-only with adaptive windows',
                    'window_sizes': {tf: config['window_size'] for tf, config in self.timeframes.items()},
                    'prediction_days': {tf: config['days'] for tf, config in self.timeframes.items()},
                    'feature_types': 'price, volume, technical indicators only'
                }
            }

            #Cache Results
            self.cache[cache_key] = (result_data, time.time())

            print(f"✅ Technical-only ML prediction analysis completed for {ticker}")
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
        
        except Exception as e:
            print(f"❌ Error in ML prediction analysis for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _prepare_technical_training_dataset(self, historical_data_result: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
        """Prepare technical-only training datasets for each timeframe with adaptive window sizes"""
        try:
            print(f"🔧 Preparing technical-only training datasets for {ticker}...")

            #Get historical data
            historical_data = historical_data_result.get('historical_data')
            if historical_data is None or historical_data.empty:
                print("❌ No historical data available")
                return None

            #Need sufficient data for the largest window (120 days) plus prediction period (60 days)
            min_required_data = 200 #At least 20 windows for long-term timeframe
            if len(historical_data) < min_required_data:
                print(f"⚠️ Insufficient historical data: {len(historical_data)} days, need at least {min_required_data}")
                return None

            #Create separate datasets for each timeframe with adaptive windows
            timeframe_datasets = {}

            for timeframe, config in self.timeframes.items():
                prediction_days = config['days']  #5, 20, or 60 days
                window_size = config['window_size']  #20, 60, or 120 days
                
                print(f"📊 Preparing {timeframe} dataset (window: {window_size} days, predicting {prediction_days} days ahead)")

                #For traditional ML models
                features_list = []
                #For LSTM models - sequential data
                lstm_sequences = []
                targets_classification = []
                targets_regression = []

                for i in range(window_size, len(historical_data) - prediction_days):
                    try:
                        #Extract technical features from the appropriate window
                        window_data = historical_data.iloc[i-window_size:i]

                        #Traditional ML features (aggregated)
                        features = self._extract_technical_features(window_data, timeframe)

                        #LSTM sequential features (daily raw + basic technical indicators)
                        lstm_sequence = self._extract_lstm_sequence(window_data, timeframe)

                        if features is None or lstm_sequence is None:
                            continue

                        #Calculate targets for THIS specific timeframe
                        current_price = historical_data.iloc[i]['Close']
                        future_price = historical_data.iloc[i + prediction_days]['Close']

                        price_change = (future_price - current_price) / current_price

                        #Adjust thresholds based on timeframe (longer periods need higher thresholds)
                        if timeframe == 'short_term':
                            buy_threshold = 0.05   #5% for 5 days
                            sell_threshold = -0.05
                        elif timeframe == 'medium_term':
                            buy_threshold = 0.10   #10% for 20 days
                            sell_threshold = -0.10
                        else: #long_term
                            buy_threshold = 0.15   #15% for 60 days
                            sell_threshold = -0.15

                        #Classification target
                        if price_change > buy_threshold:
                            target_class = 2 #BUY
                        elif price_change < sell_threshold:
                            target_class = 0 #SELL
                        else:
                            target_class = 1 #HOLD

                        features_list.append(features)
                        lstm_sequences.append(lstm_sequence)
                        targets_classification.append(target_class)
                        targets_regression.append(price_change)

                    except Exception as e:
                        print(f"⚠️ Error processing window {i} for {timeframe}: {str(e)}")
                        continue

                if len(features_list) < 20: #Need minimum samples
                    print(f"❌ Insufficient training samples for {timeframe}: {len(features_list)}")
                    continue

                #Convert to arrays and scale
                features_array = np.array(features_list)
                lstm_sequences_array = np.array(lstm_sequences)
                targets_class_array = np.array(targets_classification)
                targets_reg_array = np.array(targets_regression)

                #Scale traditional features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)

                #Scale LSTM sequences (scale each timestep's features)
                lstm_scaler = StandardScaler()
                #Reshape for scaling: (samples * timesteps, features)
                lstm_reshaped = lstm_sequences_array.reshape(-1, lstm_sequences_array.shape[-1])
                lstm_scaled_reshaped = lstm_scaler.fit_transform(lstm_reshaped)
                # Reshape back: (samples, timesteps, features)
                lstm_sequences_scaled = lstm_scaled_reshaped.reshape(lstm_sequences_array.shape)

                timeframe_datasets[timeframe] = {
                    'features': features_scaled,
                    'features_raw': features_array,
                    'lstm_sequences': lstm_sequences_scaled,
                    'lstm_sequences_raw': lstm_sequences_array,
                    'targets_classification': targets_class_array,
                    'targets_regression': targets_reg_array,
                    'scaler': scaler,
                    'lstm_scaler': lstm_scaler,
                    'samples_count': len(features_list),
                    'prediction_days': prediction_days,
                    'window_size': window_size,
                    'feature_names': self._get_technical_feature_names(timeframe),
                    'lstm_timesteps': lstm_sequences_array.shape[1],
                    'lstm_features': lstm_sequences_array.shape[2]
                }

                print(f"✅ {timeframe} dataset: {len(features_list)} samples, {features_array.shape[1]} traditional features, LSTM: {lstm_sequences_array.shape[1]} timesteps × {lstm_sequences_array.shape[2]} features")

            return timeframe_datasets

        except Exception as e:
            print(f"❌ Error preparing technical training datasets: {str(e)}")
            return None

    def _extract_lstm_sequence(self, window_data: pd.DataFrame, timeframe: str) -> Optional[np.ndarray]:
        """Extract sequential data for LSTM model with proper timestep dependence"""
        try:
            #Determine sequence length based on timeframe
            if timeframe == 'short_term':
                sequence_length = min(10, len(window_data))  #Last 10 days for short-term
            elif timeframe == 'medium_term':
                sequence_length = min(20, len(window_data))  #Last 20 days for medium-term
            else:
                sequence_length = min(30, len(window_data))  #Last 30 days for long-term
            
            #Get the most recent sequence_length days
            recent_data = window_data.tail(sequence_length)
            
            #Create sequence: each timestep has multiple features
            sequence = []
            
            for i in range(len(recent_data)):
                day_features = []
                
                #Basic OHLC data 
                if i == 0:
                    #For first day, use previous day ratios as 1.0
                    day_features.extend([1.0, 1.0, 1.0, 1.0])  #Open, High, Low, Close ratios
                else:
                    prev_close = recent_data.iloc[i-1]['Close']
                    curr_open = recent_data.iloc[i]['Open']
                    curr_high = recent_data.iloc[i]['High']
                    curr_low = recent_data.iloc[i]['Low']
                    curr_close = recent_data.iloc[i]['Close']
                    
                    #Price ratios relative to previous close
                    day_features.extend([
                        curr_open / prev_close if prev_close > 0 else 1.0,
                        curr_high / prev_close if prev_close > 0 else 1.0,
                        curr_low / prev_close if prev_close > 0 else 1.0,
                        curr_close / prev_close if prev_close > 0 else 1.0
                    ])
                
                #Volume ratio
                if 'Volume' in recent_data.columns:
                    avg_volume = recent_data['Volume'].mean()
                    volume_ratio = recent_data.iloc[i]['Volume'] / avg_volume if avg_volume > 0 else 1.0
                else:
                    volume_ratio = 1.0
                day_features.append(volume_ratio)
                
                #Daily return
                if i == 0:
                    daily_return = 0.0
                else:
                    prev_close = recent_data.iloc[i-1]['Close']
                    curr_close = recent_data.iloc[i]['Close']
                    daily_return = (curr_close - prev_close) / prev_close if prev_close > 0 else 0.0
                day_features.append(daily_return)
                
                #High-Low range (volatility proxy)
                curr_high = recent_data.iloc[i]['High']
                curr_low = recent_data.iloc[i]['Low']
                curr_close = recent_data.iloc[i]['Close']
                hl_range = (curr_high - curr_low) / curr_close if curr_close > 0 else 0.0
                day_features.append(hl_range)
                
                #Simple moving average position (if enough data)
                if i >= 4:  #Need at least 5 days for SMA5
                    sma_5 = recent_data.iloc[i-4:i+1]['Close'].mean()
                    sma_position = curr_close / sma_5 if sma_5 > 0 else 1.0
                else:
                    sma_position = 1.0
                day_features.append(sma_position)
                
                #RSI-like momentum indicator (simplified)
                if i >= 4:  #Need at least 5 days
                    recent_returns = []
                    for j in range(i-4, i+1):
                        if j > 0:
                            prev_price = recent_data.iloc[j-1]['Close']
                            curr_price = recent_data.iloc[j]['Close']
                            ret = (curr_price - prev_price) / prev_price if prev_price > 0 else 0.0
                            recent_returns.append(ret)
                    
                    if recent_returns:
                        gains = [r for r in recent_returns if r > 0]
                        losses = [abs(r) for r in recent_returns if r < 0]
                        avg_gain = np.mean(gains) if gains else 0.0
                        avg_loss = np.mean(losses) if losses else 0.0
                        rsi_like = avg_gain / (avg_gain + avg_loss) if (avg_gain + avg_loss) > 0 else 0.5
                    else:
                        rsi_like = 0.5
                else:
                    rsi_like = 0.5
                day_features.append(rsi_like)
                
                sequence.append(day_features)
            
            return np.array(sequence)  #Shape: (sequence_length, features_per_timestep)
            
        except Exception as e:
            print(f"⚠️ Error extracting LSTM sequence for {timeframe}: {str(e)}")
            return None

    def _extract_technical_features(self, window_data: pd.DataFrame, timeframe: str) -> Optional[List[float]]:
        """Extract technical features optimized for specific timeframe"""
        try:
            features = []

            #1. ALWAYS INCLUDE: Basic price/volume features
            features.extend(self._extract_basic_price_features(window_data))

            #2. ALWAYS INCLUDE: Technical indicators
            features.extend(self._extract_computed_technical_features(window_data))

            #3. TIMEFRAME-SPECIFIC TECHNICAL FEATURES
            if timeframe == 'short_term':
                #Short-term: Focus on momentum and recent patterns
                features.extend(self._extract_momentum_features(window_data))
                features.extend(self._extract_volatility_features(window_data))
                
            elif timeframe == 'medium_term':
                #Medium-term: Focus on trends and intermediate patterns
                features.extend(self._extract_trend_features(window_data))
                features.extend(self._extract_cycle_features(window_data))
                
            else:  #long_term
                #Long-term: Focus on major cycles and structural patterns
                features.extend(self._extract_structural_features(window_data))
                features.extend(self._extract_regime_features(window_data))

            return features

        except Exception as e:
            print(f"⚠️ Error extracting technical features for {timeframe}: {str(e)}")
            return None

    def _extract_basic_price_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract basic price/volume features (used by all timeframes)"""
        try:
            features = []
            closes = window_data['Close'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            volumes = window_data['Volume'].values if 'Volume' in window_data.columns else np.ones(len(closes))

            #Price momentum (multiple timeframes)
            features.extend([
                (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0.0,
                (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0.0,
                (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0.0
            ])

            #Moving averages
            sma_5 = np.mean(closes[-5:]) if len(closes) >= 5 else closes[-1]
            sma_10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]

            features.extend([
                (closes[-1] - sma_5) / sma_5,
                (closes[-1] - sma_10) / sma_10,
                (closes[-1] - sma_20) / sma_20,
                (sma_5 - sma_10) / sma_10 if sma_10 != 0 else 0.0,
                (sma_10 - sma_20) / sma_20 if sma_20 != 0 else 0.0
            ])

            #Volatility
            returns = np.diff(closes) / closes[:-1]
            features.extend([
                np.std(returns[-5:]) if len(returns) >= 5 else 0.0,
                np.std(returns[-10:]) if len(returns) >= 10 else 0.0,
                np.std(returns[-20:]) if len(returns) >= 20 else 0.0
            ])

            #Price position in range
            recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
            recent_low = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
            range_size = recent_high - recent_low

            features.extend([
                (closes[-1] - recent_low) / range_size if range_size > 0 else 0.5,
                (recent_high - closes[-1]) / recent_high if recent_high > 0 else 0.0,
                (closes[-1] - recent_low) / recent_low if recent_low > 0 else 0.0,
            ])

            #Volume patterns
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            features.extend([
                volumes[-1] / avg_volume if avg_volume > 0 else 1.0,
                np.mean(volumes[-5:]) / avg_volume if avg_volume > 0 and len(volumes) >= 5 else 1.0,
            ])

            return features  #16 basic features

        except Exception as e:
            print(f"⚠️ Error extracting basic price features: {str(e)}")
            return [0.0] * 16

    def _extract_computed_technical_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract computed technical indicators (used by all timeframes)"""
        try:
            features = []
            closes = window_data['Close'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #RSI
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses) if len(losses) > 0 else 0
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 50
            features.append(rsi / 100)

            #MACD
            ema_12_array = self._calculate_ema_array(closes, 12)
            ema_26_array = self._calculate_ema_array(closes, 26)
            macd_line = ema_12_array - ema_26_array
            macd_signal = self._calculate_ema_array(macd_line, 9)
            macd_histogram = macd_line - macd_signal

            features.extend([
                1.0 if macd_line[-1] > macd_signal[-1] else 0.0,
                np.clip(macd_histogram[-1] / 10.0, -1, 1),
            ])

            #Bollinger Bands
            sma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            bb_std = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes)
            bb_upper = sma_20 + (2 * bb_std)
            bb_lower = sma_20 - (2 * bb_std)
            bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

            features.extend([
                bb_position,
                1.0 if closes[-1] < bb_lower else 0.0,
                1.0 if closes[-1] > bb_upper else 0.0,
            ])

            #Stochastic
            lowest_low = np.min(lows[-14:]) if len(lows) >= 14 else np.min(lows)
            highest_high = np.max(highs[-14:]) if len(highs) >= 14 else np.max(highs)
            stoch_range = highest_high - lowest_low
            stoch_k = ((closes[-1] - lowest_low) / stoch_range) * 100 if stoch_range > 0 else 50

            features.extend([
                stoch_k / 100.0,
                1.0 if stoch_k < 20 else 0.0,
                1.0 if stoch_k > 80 else 0.0,
            ])

            return features  #9 technical features

        except Exception as e:
            print(f"⚠️ Error extracting computed technical features: {str(e)}")
            return [0.0] * 9

    def _extract_momentum_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract momentum features for short-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            volumes = window_data['Volume'].values if 'Volume' in window_data.columns else np.ones(len(closes))
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #Short-term momentum indicators
            features.extend([
                1.0 if np.mean(returns[-3:]) > 0 else 0.0 if len(returns) >= 3 else 0.5,  #Recent 3-day trend
                1.0 if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3] else 0.0,  #3-day uptrend
                1.0 if len(volumes) >= 5 and volumes[-1] > np.mean(volumes[-5:]) * 1.5 else 0.0,  #Volume spike
                np.mean(returns[-2:]) if len(returns) >= 2 else 0.0,  #Very recent momentum
                np.sum(returns[-5:] > 0) / 5.0 if len(returns) >= 5 else 0.5,  #Win rate last 5 days
            ])

            return features  #5 momentum features

        except Exception as e:
            print(f"⚠️ Error extracting momentum features: {str(e)}")
            return [0.0] * 5

    def _extract_volatility_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract volatility features for short-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            highs = window_data['High'].values
            lows = window_data['Low'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #Volatility measures
            features.extend([
                np.std(returns[-3:]) if len(returns) >= 3 else 0.0,  #Very short-term volatility
                np.mean([(highs[i] - lows[i]) / closes[i] for i in range(-5, 0)]) if len(closes) >= 5 else 0.0,  #Average daily range
                1.0 if len(returns) >= 5 and np.std(returns[-5:]) > np.std(returns[-20:]) else 0.0 if len(returns) >= 20 else 0.5,  #Volatility expansion
            ])

            return features  #3 volatility features

        except Exception as e:
            print(f"⚠️ Error extracting volatility features: {str(e)}")
            return [0.0] * 3

    def _extract_trend_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract trend features for medium-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #Medium-term trend indicators
            features.extend([
                1.0 if len(returns) >= 10 and np.mean(returns[-10:]) > 0 else 0.0,  #10-day trend
                1.0 if len(closes) >= 15 and np.mean(closes[-5:]) > np.mean(closes[-15:]) else 0.0,  #Trend acceleration
                1.0 if len(closes) >= 30 and closes[-1] > np.mean(closes[-30:]) else 0.0,  #Above 30-day average
                np.corrcoef(range(len(closes)), closes)[0, 1] if len(closes) > 1 else 0.0,  #Price trend correlation
            ])

            return features  #4 trend features

        except Exception as e:
            print(f"⚠️ Error extracting trend features: {str(e)}")
            return [0.0] * 4

    def _extract_cycle_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract cycle features for medium-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            
            if len(closes) >= 20:
                #Detrended Price Oscillator (DPO) - removes trend to show cycles
                period = 20
                sma_shifted = np.mean(closes[-(period//2 + 1 + period):-(period//2 + 1)]) if len(closes) >= period + period//2 else closes[-1]
                dpo = closes[-1] - sma_shifted
                dpo_normalized = dpo / closes[-1] if closes[-1] > 0 else 0.0
                
                #Cycle momentum - rate of change in cycles
                if len(closes) == 60:
                    prev_dpo = closes[-20] - np.mean(closes[-(period//2 + 1 + period + 20):-(period//2 + 1 + 20)])
                    dpo_momentum = (dpo - prev_dpo) / abs(prev_dpo) if prev_dpo != 0 else 0.0
                else:
                    dpo_momentum = 0.0
                
                #Cycle phase - where are we in the cycle?
                recent_highs = np.max(closes[-10:]) if len(closes) >= 10 else closes[-1]
                recent_lows = np.min(closes[-10:]) if len(closes) >= 10 else closes[-1]
                cycle_phase = (closes[-1] - recent_lows) / (recent_highs - recent_lows) if recent_highs != recent_lows else 0.5
                
            else:
                dpo_normalized = 0.0
                dpo_momentum = 0.0
                cycle_phase = 0.5
            
            features.extend([
                dpo_normalized,           #Detrended price position
                dpo_momentum,            #Cycle momentum
                cycle_phase,             #Current cycle phase (0=bottom, 1=top)
                1.0 if len(closes) >= 40 and closes[-1] > closes[-20] > closes[-40] else 0.0,  #Multi-period uptrend
                1.0 if dpo_normalized > 0 else 0.0,  #Above/below cycle center
            ])

            return features  #5 cycle features

        except Exception as e:
            print(f"⚠️ Error extracting cycle features: {str(e)}")
            return [0.0] * 5

    def _extract_structural_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract structural features for long-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #Long-term structural indicators
            features.extend([
                1.0 if len(returns) >= 100 and np.mean(returns[-100:]) > 0 else 0.0,  #Long-term trend
                (closes[-1] - closes[0]) / closes[0] if len(closes) > 0 and closes[0] != 0 else 0.0,  #Total window return
                np.std(returns[-100:]) if len(returns) >= 100 else np.std(returns),  #Long-term volatility
                1.0 if len(closes) >= 100 and np.mean(closes[-20:]) > np.mean(closes[-100:]) else 0.0,  #Recent vs distant average
            ])

            return features  #4 structural features

        except Exception as e:
            print(f"⚠️ Error extracting structural features: {str(e)}")
            return [0.0] * 4

    def _extract_regime_features(self, window_data: pd.DataFrame) -> List[float]:
        """Extract market regime features for long-term prediction"""
        try:
            features = []
            closes = window_data['Close'].values
            returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])

            #Market regime indicators
            features.extend([
                1.0 if len(returns) >= 30 and np.mean(returns[-30:]) > 2 * np.std(returns[-30:]) else 0.0,  #Strong bull regime
                1.0 if len(returns) >= 30 and np.mean(returns[-30:]) < -2 * np.std(returns[-30:]) else 0.0,  #Strong bear regime
                np.mean([1.0 if r > 0 else 0.0 for r in returns[-30:]]) if len(returns) >= 30 else 0.5,  #Win rate
            ])

            return features  #3 regime features

        except Exception as e:
            print(f"⚠️ Error extracting regime features: {str(e)}")
            return [0.0] * 3

    def _get_technical_feature_names(self, timeframe: str) -> List[str]:
        """Get feature names for technical features by timeframe"""
        base_names = [
            #Basic price features (16)
            'momentum_5d', 'momentum_10d', 'momentum_20d',
            'sma5_distance', 'sma10_distance', 'sma20_distance', 'sma5_vs_sma10', 'sma10_vs_sma20',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'price_position', 'distance_from_high', 'distance_from_low',
            'volume_ratio', 'avg_volume_ratio',
            
            #Technical features (9)
            'rsi', 'macd_bullish', 'macd_histogram',
            'bb_position', 'bb_below_lower', 'bb_above_upper',
            'stoch_k', 'stoch_oversold', 'stoch_overbought'
        ]

        if timeframe == 'short_term':
            base_names.extend([
                #Momentum features (5)
                'recent_3d_trend', '3d_uptrend', 'volume_spike', 'very_recent_momentum', 'win_rate_5d',
                #Volatility features (3)
                'short_volatility', 'avg_daily_range', 'volatility_expansion'
            ])
        elif timeframe == 'medium_term':
            base_names.extend([
                #Trend features (4)
                '10d_trend', 'trend_acceleration', 'above_30d_avg', 'price_trend_correlation',
                #Cycle features (5)
                'detrended_price_pos', 'cycle_momentum', 'cycle_phase', 'multi_period_uptrend', 'above_cycle_center'
            ])
        else:  #long_term
            base_names.extend([
                #Structural features (4)
                'long_term_trend', 'total_window_return', 'long_term_volatility', 'recent_vs_distant',
                #Regime features (3)
                'strong_bull_regime', 'strong_bear_regime', 'long_term_win_rate'
            ])

        return base_names

    def _calculate_ema_array(self, data, window):
        """Calculate EMA array for historical data (returns full array)"""
        if not hasattr(data, '__len__') or len(data) < window:
            return np.array([np.mean(data) if hasattr(data, '__len__') and len(data) > 0 else 0])
        
        # Convert to numpy array if it isn't already
        data = np.array(data)
        alpha = 2 / (window + 1)
        ema_array = np.zeros_like(data, dtype=float)
        ema_array[0] = data[0]
        
        for i in range(1, len(data)):
            ema_array[i] = alpha * data[i] + (1 - alpha) * ema_array[i-1]
        
        return ema_array

    def _train_models(self, timeframe_datasets: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Train separate ML models for each timeframe"""
        try:
            print(f"🤖 Training timeframe-specific technical ML models for {ticker}...")
            
            all_trained_models = {}
            
            for timeframe, training_data in timeframe_datasets.items():
                print(f"🎯 Training {timeframe} models ({training_data['prediction_days']} days ahead, {training_data['window_size']} day window)...")
                
                features = training_data['features']
                targets_class = training_data['targets_classification']
                targets_reg = training_data['targets_regression']
                
                timeframe_models = {}
                
                #Split data for validation
                X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
                    features, targets_class, test_size=0.2, random_state=42, stratify=targets_class
                )
                
                X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                    features, targets_reg, test_size=0.2, random_state=42
                )
                
                #Train classification models optimized for timeframe
                print(f"📊 Training {timeframe} classification models...")
                
                #Random Forest (always included)
                rf_classifier = RandomForestClassifier(
                    n_estimators=150 if timeframe == 'long_term' else 100,
                    max_depth=12 if timeframe == 'long_term' else 10,
                    min_samples_split=8 if timeframe == 'short_term' else 10,
                    random_state=42,
                    class_weight='balanced'
                )
                rf_classifier.fit(X_train_class, y_train_class)
                rf_class_score = accuracy_score(y_test_class, rf_classifier.predict(X_test_class))
                
                timeframe_models['classification'] = {
                    'random_forest': {'model': rf_classifier, 'score': rf_class_score}
                }
                
                #Add timeframe-specific models
                if timeframe == 'short_term':
                    #SVM for short-term (good for quick patterns)
                    svm_classifier = SVC(
                        kernel='rbf',
                        probability=True,
                        random_state=42,
                        class_weight='balanced',
                        C=1.0
                    )
                    svm_classifier.fit(X_train_class, y_train_class)
                    svm_class_score = accuracy_score(y_test_class, svm_classifier.predict(X_test_class))
                    timeframe_models['classification']['svm'] = {'model': svm_classifier, 'score': svm_class_score}
                    
                elif timeframe == 'long_term':
                    #Logistic Regression for long-term
                    lr_classifier = LogisticRegression(
                        random_state=42,
                        class_weight='balanced',
                        max_iter=2000,
                        C=0.1
                    )
                    lr_classifier.fit(X_train_class, y_train_class)
                    lr_class_score = accuracy_score(y_test_class, lr_classifier.predict(X_test_class))
                    timeframe_models['classification']['logistic_regression'] = {'model': lr_classifier, 'score': lr_class_score}
                
                #Train regression models
                print(f"📈 Training {timeframe} regression models...")
                
                #Random Forest Regressor (always included)
                rf_regressor = RandomForestRegressor(
                    n_estimators=150 if timeframe == 'long_term' else 100,
                    max_depth=12 if timeframe == 'long_term' else 10,
                    min_samples_split=8 if timeframe == 'short_term' else 10,
                    random_state=42
                )
                rf_regressor.fit(X_train_reg, y_train_reg)
                rf_reg_score = -mean_squared_error(y_test_reg, rf_regressor.predict(X_test_reg))
                
                timeframe_models['regression'] = {
                    'random_forest': {'model': rf_regressor, 'score': rf_reg_score}
                }
                
                #Add timeframe-specific regression models
                if timeframe == 'short_term':
                    #SVR for short-term
                    svr_regressor = SVR(kernel='rbf', C=1.0, epsilon=0.01)
                    svr_regressor.fit(X_train_reg, y_train_reg)
                    svr_reg_score = -mean_squared_error(y_test_reg, svr_regressor.predict(X_test_reg))
                    timeframe_models['regression']['svr'] = {'model': svr_regressor, 'score': svr_reg_score}
                    
                elif timeframe in ['medium_term', 'long_term']:
                    #Gradient Boosting for medium and long-term
                    gb_regressor = GradientBoostingRegressor(
                        n_estimators=150 if timeframe == 'long_term' else 100,
                        learning_rate=0.05 if timeframe == 'long_term' else 0.1,
                        max_depth=8 if timeframe == 'long_term' else 6,
                        random_state=42
                    )
                    gb_regressor.fit(X_train_reg, y_train_reg)
                    gb_reg_score = -mean_squared_error(y_test_reg, gb_regressor.predict(X_test_reg))
                    timeframe_models['regression']['gradient_boosting'] = {'model': gb_regressor, 'score': gb_reg_score}
                
                #Train LSTM if TensorFlow is available and sufficient data
                if TENSORFLOW_AVAILABLE:
                    try:
                        #Get LSTM sequential data
                        lstm_sequences = training_data['lstm_sequences']
                        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                            lstm_sequences, targets_reg, test_size=0.2, random_state=42
                        )
                        
                        lstm_model = self._train_timeframe_lstm(X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, timeframe)
                        if lstm_model:
                            timeframe_models['regression']['lstm'] = lstm_model
                    except Exception as e:
                        print(f"⚠️ Error training {timeframe} LSTM: {str(e)}")
                
                #Store timeframe models with metadata
                all_trained_models[timeframe] = {
                    'models': timeframe_models,
                    'scaler': training_data['scaler'],
                    'lstm_scaler': training_data.get('lstm_scaler'),
                    'feature_names': training_data['feature_names'],
                    'training_samples': training_data['samples_count'],
                    'prediction_days': training_data['prediction_days'],
                    'window_size': training_data['window_size'],
                    'lstm_timesteps': training_data.get('lstm_timesteps'),
                    'lstm_features': training_data.get('lstm_features')
                }
                
                #Print scores for this timeframe
                class_scores = [f"{name}={data['score']:.3f}" for name, data in timeframe_models['classification'].items()]
                reg_scores = [f"{name}={data['score']:.6f}" for name, data in timeframe_models['regression'].items()]
                print(f"✅ {timeframe} training completed:")
                print(f"   📊 Classification: {', '.join(class_scores)}")
                print(f"   📈 Regression: {', '.join(reg_scores)}")
                print(f"   📝 Features: {len(training_data['feature_names'])}, Samples: {training_data['samples_count']}")
            
            print(f"✅ Technical-only model training completed. Trained models for {len(all_trained_models)} timeframes.")
            return all_trained_models
            
        except Exception as e:
            print(f"❌ Error training models: {str(e)}")
            return {}

    def _train_timeframe_lstm(self, X_train, X_test, y_train, y_test, timeframe):
        """Train LSTM model optimized for specific timeframe with proper sequential data"""
        try:
            print(f"🧠 Training {timeframe} LSTM model with sequential data...")
            print(f"   📊 Input shape: {X_train.shape} (samples, timesteps, features)")

            #X_train is already in the correct shape: (samples, timesteps, features)
            timesteps = X_train.shape[1]
            features = X_train.shape[2]

            #Build timeframe-optimized LSTM model
            if timeframe == 'short_term':
                #Smaller, faster LSTM for short-term patterns (10 timesteps)
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                epochs = 50
                learning_rate = 0.001
                
            elif timeframe == 'medium_term':
                #Balanced LSTM for medium-term patterns (20 timesteps)
                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
                    Dropout(0.3),
                    LSTM(64, return_sequences=True),
                    Dropout(0.3),
                    LSTM(32, return_sequences=False),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                epochs = 75
                learning_rate = 0.0008
                
            else:
                return None

            #Compile with timeframe-appropriate settings
            model.compile(
                optimizer=Adam(learning_rate=learning_rate), 
                loss='mse', 
                metrics=['mae']
            )

            print(f"   🏗️ Model architecture: {timesteps} timesteps, {features} features per timestep")
            print(f"   ⚙️ Training: {epochs} epochs, lr={learning_rate}")
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
            ]

            history = model.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=32, 
                validation_data=(X_test, y_test), 
                verbose=0,
                callbacks=callbacks
            )

            #Evaluate
            train_pred = model.predict(X_train, verbose=0)
            test_pred = model.predict(X_test, verbose=0)

            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)

            print(f"✅ {timeframe} LSTM - Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            print(f"   📈 Training completed in {len(history.history['loss'])} epochs")

            return {
                'model': model,
                'score': -test_mse,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'timesteps': timesteps,
                'features_per_timestep': features,
                'epochs_trained': len(history.history['loss'])
            }

        except Exception as e:
            print(f"⚠️ Error training {timeframe} LSTM: {str(e)}")
            return None

    def _generate_timeframe_prediction(self, timeframe: str, current_data: Dict[str, Any], timeframe_models: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for specific timeframe using trained models"""
        try:
            print(f"🔮 Generating {timeframe} prediction...")
            
            prediction_data = {
                'timeframe': timeframe,
                'prediction_days': timeframe_models.get('prediction_days', self.timeframes[timeframe]['days']),
                'window_size': timeframe_models.get('window_size', self.timeframes[timeframe]['window_size']),
                'prediction_score': 0.0,
                'prediction_class': 'HOLD',
                'confidence': 0.0,
                'expected_return': 0.0,
                'signals': []
            }
            
            if not timeframe_models.get('models'):
                prediction_data['signals'].append("No trained models available")
                return prediction_data
            
            #Extract current features for this timeframe
            current_features = self._extract_current_timeframe_features(current_data, timeframe, timeframe_models['window_size'])
            if current_features is None:
                prediction_data['signals'].append("Error: Could not extract features")
                return prediction_data
            
            #Extract current LSTM sequence for this timeframe
            current_lstm_sequence = self._extract_current_lstm_sequence(current_data, timeframe, timeframe_models['window_size'])
            
            #Scale features using the timeframe's scaler
            scaler = timeframe_models.get('scaler')
            if scaler:
                current_features_scaled = scaler.transform([current_features])[0]
            else:
                current_features_scaled = current_features
            
            #Scale LSTM sequence using the timeframe's LSTM scaler
            lstm_scaler = timeframe_models.get('lstm_scaler')
            if lstm_scaler and current_lstm_sequence is not None:
                #Reshape for scaling: (timesteps * features,)
                lstm_reshaped = current_lstm_sequence.reshape(-1, current_lstm_sequence.shape[-1])
                lstm_scaled_reshaped = lstm_scaler.transform(lstm_reshaped)
                #Reshape back: (timesteps, features)
                current_lstm_scaled = lstm_scaled_reshaped.reshape(current_lstm_sequence.shape)
            else:
                current_lstm_scaled = current_lstm_sequence
            
            #Get predictions from all models for this timeframe
            model_predictions = []
            model_confidences = []
            models = timeframe_models['models']
            
            #Classification models
            if 'classification' in models:
                for model_name, model_info in models['classification'].items():
                    try:
                        model = model_info['model']
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba([current_features_scaled])[0]
                            prediction_score = np.argmax(probabilities)  #0=SELL, 1=HOLD, 2=BUY
                            confidence = np.max(probabilities)
                        else:
                            prediction = model.predict([current_features_scaled])[0]
                            prediction_score = int(prediction)
                            confidence = 0.7  #Default confidence for non-probabilistic models
                        
                        model_predictions.append(prediction_score)
                        model_confidences.append(confidence)
                        
                        class_names = ['SELL', 'HOLD', 'BUY']
                        prediction_data['signals'].append(f"{model_name}_class: {class_names[prediction_score]} ({confidence:.3f})")
                        
                    except Exception as e:
                        print(f"⚠️ Error with {model_name} classifier: {str(e)}")
                        continue
            
            #Regression models
            if 'regression' in models:
                for model_name, model_info in models['regression'].items():
                    try:
                        model = model_info['model']
                        
                        if model_name == 'lstm':
                            #LSTM needs sequential data
                            if current_lstm_scaled is not None:
                                #Reshape for batch prediction: (1, timesteps, features)
                                lstm_input = np.expand_dims(current_lstm_scaled, axis=0)
                                prediction_value = model.predict(lstm_input, verbose=0)[0][0]
                            else:
                                print(f"⚠️ No LSTM sequence available for {model_name}")
                                continue
                        else:
                            prediction_value = model.predict([current_features_scaled])[0]
                    
                        #Convert regression to classification based on timeframe thresholds
                        if timeframe == 'short_term':
                            buy_threshold, sell_threshold = 0.05, -0.05
                        elif timeframe == 'medium_term':
                            buy_threshold, sell_threshold = 0.10, -0.10
                        else:  #long_term
                            buy_threshold, sell_threshold = 0.15, -0.15
                        
                        if prediction_value > buy_threshold:
                            prediction_score = 2  #BUY
                        elif prediction_value < sell_threshold:
                            prediction_score = 0  #SELL
                        else:
                            prediction_score = 1  #HOLD
                        
                        #Confidence based on magnitude relative to threshold
                        confidence = min(abs(prediction_value) / max(abs(buy_threshold), abs(sell_threshold)), 1.0)
                        
                        model_predictions.append(prediction_score)
                        model_confidences.append(confidence)
                        
                        prediction_data['signals'].append(f"{model_name}_reg: {prediction_value:.3f} return ({confidence:.3f})")
                        
                    except Exception as e:
                        print(f"⚠️ Error with {model_name} regressor: {str(e)}")
                        continue
            
            if model_predictions:
                #Ensemble prediction (weighted average)
                weights = np.array(model_confidences)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(model_predictions)) / len(model_predictions)
                
                #Weighted average of predictions
                weighted_prediction = np.average(model_predictions, weights=weights)
                average_confidence = np.mean(model_confidences)
                
                #Convert to final classification
                if weighted_prediction > 1.5:
                    final_class = 'BUY'
                elif weighted_prediction < 0.5:
                    final_class = 'SELL'
                else:
                    final_class = 'HOLD'
                
                #Calculate expected return based on timeframe and confidence
                if final_class == 'BUY':
                    base_return = 0.05 if timeframe == 'short_term' else (0.10 if timeframe == 'medium_term' else 0.15)
                    expected_return = base_return * average_confidence
                elif final_class == 'SELL':
                    base_return = -0.05 if timeframe == 'short_term' else (-0.10 if timeframe == 'medium_term' else -0.15)
                    expected_return = base_return * average_confidence
                else:
                    expected_return = 0.0
                
                prediction_data.update({
                    'prediction_score': weighted_prediction,
                    'prediction_class': final_class,
                    'confidence': average_confidence,
                    'expected_return': expected_return
                })
                
                #Add ensemble signal
                prediction_data['signals'].append(f"Ensemble: {final_class} (confidence: {average_confidence:.3f}, return: {expected_return:.3f})")
            
            return prediction_data
            
        except Exception as e:
            print(f"❌ Error generating {timeframe} prediction: {str(e)}")
            return {
                'timeframe': timeframe,
                'prediction_days': self.timeframes[timeframe]['days'],
                'window_size': self.timeframes[timeframe]['window_size'],
                'prediction_score': 0.0,
                'prediction_class': 'HOLD',
                'confidence': 0.0,
                'expected_return': 0.0,
                'signals': [f"Error: {str(e)}"]
            }

    def _default_prediction(self, timeframe: str) -> Dict[str, Any]:
        """Create default prediction when models fail"""
        return {
            'timeframe': timeframe,
            'prediction_days': self.timeframes[timeframe]['days'],
            'window_size': self.timeframes[timeframe]['window_size'],
            'prediction_score': 1.0,
            'prediction_class': 'HOLD',
            'confidence': 0.0,
            'expected_return': 0.0,
            'signals': ['Default prediction - model training failed']
        }

    def _extract_current_timeframe_features(self, current_data: Dict[str, Any], timeframe: str, window_size: int) -> Optional[List[float]]:
        """Extract features for current prediction using EXACTLY the same features as training"""
        try:
            #Get historical data for feature calculation
            historical_data = current_data.get('historical_data')
            if historical_data is None or historical_data.empty:
                print("❌ No historical data for feature extraction")
                return None
            
            #Get the most recent window for feature extraction (same size as training)
            window_data = historical_data.tail(window_size)
            
            #CRITICAL: Use the EXACT SAME feature extraction method as training
            features = self._extract_technical_features(window_data, timeframe)
            
            if features is None:
                print(f"❌ Failed to extract {timeframe} technical features")
                return None
            
            print(f"✅ Extracted {len(features)} technical features for {timeframe} prediction")
            return features
            
        except Exception as e:
            print(f"❌ Error extracting current {timeframe} features: {str(e)}")
            return None

    def _extract_current_lstm_sequence(self, current_data: Dict[str, Any], timeframe: str, window_size: int) -> Optional[np.ndarray]:
        """Extract LSTM sequence for current prediction using EXACTLY the same method as training"""
        try:
            #Get historical data for sequence extraction
            historical_data = current_data.get('historical_data')
            if historical_data is None or historical_data.empty:
                print("❌ No historical data for LSTM sequence extraction")
                return None
            
            #Get the most recent window for sequence extraction (same size as training)
            window_data = historical_data.tail(window_size)
            
            #CRITICAL: Use the EXACT SAME sequence extraction method as training
            lstm_sequence = self._extract_lstm_sequence(window_data, timeframe)
            
            if lstm_sequence is None:
                print(f"❌ Failed to extract {timeframe} LSTM sequence")
                return None
            
            print(f"✅ Extracted LSTM sequence for {timeframe} prediction: {lstm_sequence.shape}")
            return lstm_sequence
            
        except Exception as e:
            print(f"❌ Error extracting current {timeframe} LSTM sequence: {str(e)}")
            return None

    def _generate_ensemble_prediction(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ensemble prediction from all timeframes"""
        try:
            #Weighted ensemble based on timeframe weights and confidences
            total_weight = 0
            weighted_score = 0
            weighted_confidence = 0
            
            for timeframe, prediction in predictions.items():
                if prediction and prediction.get('confidence', 0) > 0:
                    timeframe_weight = self.timeframes[timeframe]['weight']
                    confidence_weight = prediction['confidence']
                    combined_weight = timeframe_weight * confidence_weight
                    
                    weighted_score += prediction['prediction_score'] * combined_weight
                    weighted_confidence += prediction['confidence'] * combined_weight
                    total_weight += combined_weight
            
            if total_weight > 0:
                ensemble_score = weighted_score / total_weight
                ensemble_confidence = weighted_confidence / total_weight
            else:
                ensemble_score = 1.0  # HOLD
                ensemble_confidence = 0.0
            
            #Convert to final classification
            if ensemble_score > 1.5:
                final_class = 'BUY'
                expected_return = 0.08 * ensemble_confidence  #Average expected return
            elif ensemble_score < 0.5:
                final_class = 'SELL'
                expected_return = -0.08 * ensemble_confidence
            else:
                final_class = 'HOLD'
                expected_return = 0.0
            
            return {
                'ensemble_score': ensemble_score,
                'ensemble_class': final_class,
                'ensemble_confidence': ensemble_confidence,
                'expected_return': expected_return,
                'contributing_timeframes': list(predictions.keys())
            }
            
        except Exception as e:
            print(f"⚠️ Error generating ensemble prediction: {str(e)}")
            return {
                'ensemble_score': 1.0,
                'ensemble_class': 'HOLD',
                'ensemble_confidence': 0.0,
                'expected_return': 0.0,
                'contributing_timeframes': []
            }

    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall prediction confidence"""
        try:
            confidences = []
            
            #Collect individual confidences
            for timeframe, prediction in predictions.items():
                if 'confidence' in prediction:
                    confidences.append(prediction['confidence'])
            
            #Check agreement between timeframes
            classes = [p['prediction_class'] for p in predictions.values() if 'prediction_class' in p]
            if classes:
                most_common_class = max(set(classes), key=classes.count)
                agreement_rate = classes.count(most_common_class) / len(classes)
            
            #Calculate overall confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            overall_confidence = avg_confidence * agreement_rate
            
            return {
                'overall_confidence': overall_confidence,
                'individual_confidences': dict(zip(predictions.keys(), confidences)),
                'agreement_rate': agreement_rate,
                'confidence_factors': {
                    'model_certainty': avg_confidence,
                    'timeframe_agreement': agreement_rate,
                    'data_quality': self._assess_data_quality()
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating confidence: {str(e)}")
            return {
                'overall_confidence': 0.0,
                'individual_confidences': {},
                'agreement_rate': 0.0,
                'confidence_factors': {}
            }

    def _assess_prediction_risk(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess prediction risk factors"""
        try:
            risk_factors = []
            risk_score = 0.0
            
            #Check for conflicting predictions
            classes = [p['prediction_class'] for p in predictions.values() if 'prediction_class' in p]
            if len(set(classes)) > 1:
                risk_factors.append("Conflicting timeframe predictions")
                risk_score += 0.3
            
            #Check confidence levels
            confidences = [p['confidence'] for p in predictions.values() if 'confidence' in p]
            if np.mean(confidences) < 0.6:
                risk_factors.append("Low model confidence")
                risk_score += 0.2
            
            #Check data quality
            data_quality = self._assess_data_quality()
            if data_quality < 0.9:
                risk_factors.append("Limited data quality")
                risk_score += 0.1
            
            
            risk_level = 'HIGH' if risk_score > 0.5 else 'MEDIUM' if risk_score > 0.2 else 'LOW'
            
            return {
                'risk_score': min(risk_score, 1.0),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': self._get_risk_recommendations(risk_level, risk_factors)
            }
            
        except Exception as e:
            print(f"⚠️ Error assessing risk: {str(e)}")
            return {
                'risk_score': 0.5,
                'risk_level': 'MEDIUM',
                'risk_factors': ['Risk assessment error'],
                'recommendations': []
            }

    def _get_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        
        if risk_level == 'HIGH':
            recommendations.extend([
                "Consider reducing position size",
                "Use tight stop-losses",
                "Monitor positions closely"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Use standard risk management",
                "Consider partial position sizing"
            ])
        else:
            recommendations.append("Standard risk management applies")
        
        #Specific recommendations based on risk factors
        if "Conflicting timeframe predictions" in risk_factors:
            recommendations.append("Wait for clearer signals across timeframes")
        
        if "Low model confidence" in risk_factors:
            recommendations.append("Seek additional confirmation signals")
        
        if "High market volatility" in risk_factors:
            recommendations.append("Adjust for increased volatility risk")
        
        return recommendations

    def _generate_ml_trading_signals(self, ensemble_result: Dict[str, Any], confidence_analysis: Dict[str, Any]) -> List[str]:
        """Generate trading signals based on ML predictions"""
        signals = []
        
        prediction_class = ensemble_result.get('ensemble_class', 'HOLD')
        confidence = confidence_analysis.get('overall_confidence', 0.0)
        expected_return = ensemble_result.get('expected_return', 0.0)
        
        #Main signal
        if prediction_class == 'BUY' and confidence > 0.6:
            signals.append(f"STRONG BUY: High confidence ({confidence:.2f}) technical ML prediction")
        elif prediction_class == 'BUY' and confidence > 0.4:
            signals.append(f"BUY: Moderate confidence ({confidence:.2f}) technical ML prediction")
        elif prediction_class == 'SELL' and confidence > 0.6:
            signals.append(f"STRONG SELL: High confidence ({confidence:.2f}) technical ML prediction")
        elif prediction_class == 'SELL' and confidence > 0.4:
            signals.append(f"SELL: Moderate confidence ({confidence:.2f}) technical ML prediction")
        else:
            signals.append(f"HOLD: {prediction_class} signal with {confidence:.2f} confidence")
        
        #Expected return signal
        if abs(expected_return) > 0.05:
            signals.append(f"Expected return: {expected_return:.1%} over prediction horizon")
        
        #Confidence qualifier
        if confidence < 0.4:
            signals.append("LOW CONFIDENCE: Consider additional analysis before acting")
        
        return signals

    def _get_model_performance(self) -> Dict[str, Any]:
        """Get model performance summary"""
        return {
            'last_training': datetime.now().isoformat(),
            'approach': 'technical_only',
            'performance_note': 'Technical indicators only - no fundamental or sentiment data leakage'
        }

    def _get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get feature importance summary"""
        return {
            'feature_types': [
                'Price momentum indicators',
                'Moving average relationships',
                'Volatility measures',
                'Technical indicators (RSI, MACD, Bollinger Bands, Stochastic)',
                'Volume patterns',
                'Timeframe-specific technical features'
            ],
            'excluded_features': [
                'Fundamental ratios (P/E, P/B, etc.)',
                'Financial statement data',
                'Growth metrics',
                'Profitability ratios',
                'Sentiment indicators',
                'News sentiment',
                'Social Media sentiment'
            ],
            'approach': 'Pure technical analysis to avoid data leakage'
        }

    def _assess_data_quality(self) -> float:
        """Assess overall data quality"""
        #Since we're self-contained and extract features directly from price data, data quality is consistently high
        quality_score = 0.95
        return quality_score


#Global instance
ml_prediction_tool = MLPredictionTool()