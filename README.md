# AI Stock Analysis Platform

A sophisticated full-stack application providing AI-powered stock analysis with secure API key management and real-time trading signals. This platform combines advanced machine learning, technical analysis, fundamental analysis, and sentiment analysis to deliver comprehensive investment insights.

## 🚀 Core Features

- **Multi-Timeframe AI Analysis**: Short-term (5-20 days), medium-term (20-60 days), and long-term (60+ days) trading signals
- **Advanced ML Pipeline**: Ensemble of Random Forest, Gradient Boosting, SVM, and LSTM neural networks
- **Comprehensive Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, and custom momentum indicators
- **Fundamental Analysis Engine**: Valuation metrics, financial health ratios, growth analysis, and dividend evaluation
- **Real-time Sentiment Analysis**: News sentiment, social media analysis, analyst ratings, and market sentiment indicators
- **Secure API Key Management**: SHA-256 hashed storage with granular permissions and usage tracking
- **Multi-Source Data Integration**: Yahoo Finance and Alpha Vantage with automatic fallback mechanisms
- **LangChain AI Orchestration**: Intelligent tool selection and context-aware analysis coordination

## 🏗️ Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js 15)                    │
├─────────────────────────────────────────────────────────────┤
│  • React 19 with TypeScript                                 │
│  • Tailwind CSS + Radix UI Components                       │
│  • Supabase Authentication & Database                       │
│  • API Key Management Interface                             │
│  • Real-time Demo Section                                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│  • Python 3.9+ with Async/Await                             │
│  • LangChain AI Orchestration                               │
│  • OpenAI GPT-4 Integration                                 │
│  • Multi-Source Data Pipeline                               │
│  • Caching Layer (In-Memory)                                │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Data Sources                                 │
├─────────────────────────────────────────────────────────────┤
│  • Yahoo Finance API (Primary)                              │
│  • Alpha Vantage API (Fallback)                             │
│  • News APIs for Sentiment                                  │
│  • Social Media APIs                                        │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                AI Analysis Engine                            │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │ Technical   │ │ Fundamental │ │ Sentiment   │ │   ML    │ │
│  │ Analysis    │ │ Analysis    │ │ Analysis    │ │Prediction │ 
│  │ Tool        │ │ Tool        │ │ Tool        │ │  Tool   │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

**Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS, Supabase  
**Backend**: FastAPI, Python, LangChain, OpenAI GPT-4, scikit-learn, TensorFlow

## 🧠 AI Implementation Deep Dive

### LangChain Agent Architecture

The core of our AI system is built around LangChain's agent framework, which orchestrates multiple specialized analysis tools:

```python
class StockAnalysisAgent:
    """
    Comprehensive AI Stock Analysis Agent using LangChain.
    
    Orchestrates multiple specialized analysis tools:
    - Technical Analysis
    - Fundamental Analysis  
    - Sentiment Analysis
    - ML Predictions
    
    Provides intelligent, context-aware stock analysis and recommendations.
    """
```

#### Key Design Decisions:

1. **Low Temperature (0.1)**: Ensures consistent, deterministic analysis results
2. **Conversation Memory**: Maintains context across tool interactions
3. **Function Calling**: Uses OpenAI's function calling for precise tool selection
4. **Structured Output**: Enforces consistent response format across all timeframes

### AI Tool Orchestration

The LangChain agent intelligently selects and coordinates tools based on the analysis requirements:

```python
# Tool selection prompt
system_prompt = """
You are an expert financial analyst AI agent. Your task is to analyze stocks comprehensively 
across multiple timeframes using specialized analysis tools.

Available tools:
- technical_analysis: Technical indicators, price patterns, momentum analysis
- fundamental_analysis: Financial health, valuation metrics, growth analysis  
- sentiment_analysis: News sentiment, social media, analyst ratings
- ml_prediction: Machine learning predictions using ensemble models

For each timeframe (short/medium/long), use the appropriate combination of tools to generate:
1. Signal: BUY, SELL, or HOLD
2. Confidence: 0.0 to 1.0
3. Reasoning: Detailed explanation of the analysis
"""
```

## 🔧 Technical Analysis Engine

### Implementation Philosophy

Technical analysis tool uses pure pandas/numpy implementation for maximum reliability and performance:

```python
class TechnicalAnalysisTool:
    """
    Comprehensive technical analysis tool using pure pandas/numpy implementation
    """
```

#### Key Technical Indicators Implemented:

1. **Momentum Indicators**:
   - RSI (Relative Strength Index) with multiple timeframes
   - Stochastic Oscillator

2. **Trend Indicators**:
   - Moving Averages (SMA, EMA, WMA) with multiple periods
   - MACD (Moving Average Convergence Divergence)

3. **Volatility Indicators**:
   - Bollinger Bands with dynamic periods

4. **Volume Indicators**:
   - On-Balance Volume (OBV)
   - Volume Rate of Change

5. **Custom Indicators**:
   - Momentum Score (composite momentum measure)
   - Trend Strength Score (composite trend measure)
   - Volatility Score (normalized volatility measure)

#### Advanced Features:

- **Adaptive Periods**: Indicators automatically adjust based on market volatility
- **Multi-Timeframe Analysis**: Each indicator calculated across multiple timeframes
- **Signal Generation**: Sophisticated logic combining multiple indicators
- **Caching**: cache for performance optimization

## 🤖 Machine Learning Pipeline

### ML Architecture Decisions

ML implementation focuses on **technical-only features** for several strategic reasons:

1. **Data Consistency**: Historical fundamental data is difficult to obtain and align temporally
2. **Real-time Performance**: Technical features can be calculated in real-time
3. **Proven Effectiveness**: Technical analysis has demonstrated market effectiveness
4. **Reduced Complexity**: Avoids the complexity of multi-modal feature engineering

### Ensemble Model Architecture

```python
class MLPredictionTool:
    """
    Technical-Only ML Prediction Tool using ensemble of proven models:
    - Random Forest (Classification & Regression)
    - Gradient Boosting
    - Support Vector Machines
    - LSTM Neural Networks (if TensorFlow available)
    - Linear Models
    """
```

#### Model Ensemble Strategy:

1. **Random Forest**: 
   - Classification for BUY/SELL/HOLD signals
   - Regression for price prediction
   - Handles non-linear relationships well

2. **Gradient Boosting**:
   - Sequential learning for complex patterns
   - Excellent for feature importance analysis
   - Robust to outliers

3. **Support Vector Machines**:
   - Effective for high-dimensional data
   - Good generalization properties
   - Handles non-linear relationships with kernels

4. **LSTM Neural Networks**:
   - Captures temporal dependencies
   - Sequential pattern recognition
   - Optional (requires TensorFlow)

5. **Linear Models**:
   - Baseline performance
   - Interpretable results
   - Fast training and prediction

#### Feature Engineering:

```python
# Technical features used for ML
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
```

#### Temporal Data Handling:

- **Adaptive Window Sizes**: Training windows adjust based on market volatility
- **Proper Temporal Separation**: Ensures no data leakage between train/test
- **Rolling Predictions**: Models retrain periodically with new data
- **Cross-Validation**: Time-series aware validation to prevent look-ahead bias

## 📊 Fundamental Analysis Engine

### Comprehensive Financial Analysis

```python
class FundamentalAnalysisTool:
    """
    Comprehensive fundamental analysis tool for investment decision making.
    Analyzes financial health, valuation, growth, profitability, and competitive position.
    """
```

#### Analysis Categories and Weights:

1. **Valuation (25%)**:
   - P/E Ratio (Price-to-Earnings)
   - P/B Ratio (Price-to-Book)
   - P/S Ratio (Price-to-Sales)
   - PEG Ratio (Price/Earnings to Growth)
   - EV/EBITDA (Enterprise Value to EBITDA)

2. **Financial Health (25%)**:
   - Debt-to-Equity Ratio
   - Current Ratio
   - Quick Ratio
   - ROE (Return on Equity)
   - ROA (Return on Assets)

3. **Growth (20%)**:
   - Revenue Growth (YoY, QoQ)
   - Earnings Growth
   - EPS Growth
   - Book Value Growth

4. **Profitability (15%)**:
   - Gross Margin
   - Operating Margin
   - Net Margin
   - EBITDA Margin

5. **Dividends (10%)**:
   - Dividend Yield
   - Payout Ratio
   - Dividend Growth Rate
   - Dividend Coverage

6. **Market Position (5%)**:
   - Market Capitalization
   - Beta (volatility measure)
   - Industry Comparison

## 📈 Sentiment Analysis Engine

### Multi-Source Sentiment Analysis

```python
class SentimentAnalysisTool:
    """
    Comprehensive market sentiment analysis tool for investment decision making.
    Analyzes sentiment from news, social media, analyst ratings, and market indicators.
    """
```

#### Sentiment Categories and Weights:

1. **News Sentiment (30%)**:
   - Financial news analysis
   - Earnings reports sentiment
   - Industry news impact
   - Keyword-based sentiment scoring

2. **Social Sentiment (25%)**:
   - Social media sentiment
   - Reddit discussions
   - Twitter sentiment
   - Community sentiment trends

3. **Analyst Sentiment (20%)**:
   - Analyst ratings and upgrades/downgrades
   - Price target changes
   - Earnings estimate revisions
   - Institutional sentiment

4. **Market Sentiment (25%)**:
   - Options flow analysis
   - Short interest changes
   - Insider trading patterns
   - Market breadth indicators

## 🔄 Multi-Source Data Pipeline

### Data Source Strategy

```python
class MultiSourceDataTool:
    """Multi-source data tool with fallback options"""
    
    def __init__(self):
        self.sources = [
            ("yfinance", market_data_tool),      # Primary source
            ("alpha_vantage", alpha_vantage_tool) # Fallback source
        ]
```

#### Design Decisions:

1. **Yahoo Finance Primary**: Free, reliable, comprehensive data
2. **Alpha Vantage Fallback**: Premium data source for redundancy
3. **Automatic Failover**: Seamless switching between sources
4. **Caching Strategy**: Reduces API calls and improves performance

## 📋 Prerequisites

- **Node.js 18+**: Modern JavaScript runtime
- **Python 3.9+**: Backend runtime environment
- **Supabase Account**: Database and authentication
- **OpenAI API Key**: AI analysis capabilities
- **Alpha Vantage API Key**: Financial data (optional)
- **News API Key**: Financial news data (optional)
- **Twitter bearer token**: Twitter(X) post data (optional)


## ⚡ Quick Start

### 1. Clone and Setup Backend

```bash
git clone https://github.com/yourusername/ai-stock-analysis-platform.git
cd ai-stock-analysis-platform/ai-stock-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Run backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Setup Frontend

```bash
cd ../ai-stock-frontend

# Install dependencies
npm install

# Create .env.local with your Supabase credentials
echo "NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here" > .env.local
echo "NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here" >> .env.local
echo "SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here" >> .env.local

# Run frontend
npm run dev
```

### 3. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🔐 Security Implementation

### API Key Security

**Hashing Strategy**:
```python
# SHA-256 hashing for API keys
import hashlib
import secrets

def hash_api_key(api_key: str) -> str:
    salt = secrets.token_hex(16)
    return hashlib.sha256((api_key + salt).encode()).hexdigest()
```

**Security Features**:
- **SHA-256 Hashing**: Irreversible key storage
- **Salt Generation**: Unique salt per key
- **Prefix Display**: Only first 12 characters shown
- **One-time Display**: Full keys only shown during creation
- **Usage Tracking**: Monitor API key activity
- **Instant Revocation**: Immediate deactivation capability

### Authentication Flow

1. **Supabase Auth**: JWT-based authentication
2. **API Key Validation**: Server-side key verification
3. **Permission Checking**: Granular access control
4. **CORS Configuration**: Secure cross-origin requests

## 📊 Performance Optimizations

### Caching Strategy

**Multi-Level Caching**:
- **Analysis Results**: 5-minute cache for technical analysis
- **Fundamental Data**: 1-hour cache for fundamental analysis
- **Sentiment Data**: 30-minute cache for sentiment analysis
- **ML Predictions**: 10-minute cache for ML results

**Cache Management**:
```python
def _is_cache_valid(self, cache_key: str) -> bool:
    """Check if cached data is still valid"""
    if cache_key not in self.cache:
        return False
    
    cache_data = self.cache[cache_key]
    age = time.time() - cache_data['timestamp']
    return age < self.cache_ttl
```

### Database Optimization

**Supabase Features**:
- **Row Level Security**: User data isolation
- **Real-time Subscriptions**: Live data updates
- **Automatic Scaling**: Handles traffic spikes
- **Connection Pooling**: Efficient database connections

## 🔌 API Usage

### Get Trading Signals

```bash
curl -X POST -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}' \
     "http://localhost:8000/ai-agent/signals"
```

### Response Example

```json
{
  "ticker": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "signals": {
    "short_term": {
      "signal": "BUY",
      "confidence": 0.85,
      "reasoning": "Technical analysis shows strong momentum with RSI at 45 indicating oversold conditions. ML prediction suggests 78% probability of upward movement in next 5-20 days. Recent earnings beat and positive analyst upgrades support bullish sentiment."
    },
    "medium_term": {
      "signal": "HOLD", 
      "confidence": 0.62,
      "reasoning": "Mixed signals with consolidation patterns suggesting neutral outlook. Fundamental analysis shows fair valuation with P/E ratio of 28.5. Sentiment analysis indicates moderate bullishness but concerns about market volatility."
    },
    "long_term": {
      "signal": "BUY",
      "confidence": 0.73,
      "reasoning": "Strong fundamental metrics with consistent revenue growth of 8% YoY and expanding margins. Market position remains strong with increasing market share in key segments. Long-term technical trends support positive outlook despite current market uncertainty."
    }
  },
  "execution_time_seconds": 45.2,
  "tools_used": ["technical_analysis", "ml_prediction", "sentiment_analysis", "fundamental_analysis"],
  "success": true
}
```

## 📊 Key Endpoints

- `POST /ai-agent/signals` - Generate AI trading signals (requires API key)
- `POST /ai-agent/signals-demo` - Demo trading signals (no auth required)
- `GET /health` - Check that backend is running


## 🛡️ Security Features

- **SHA-256 Hashing**: API keys are hashed before storage
- **Prefix Display**: Only first 12 characters shown for identification
- **One-time Display**: Full keys only shown during creation
- **User Isolation**: Users can only access their own keys

## 📝 Development

### Frontend Commands
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run test         # Run tests
npm run lint         # Run ESLint
```

### Backend Commands
```bash
python -m uvicorn app.main:app --reload    # Start development server
python -m pytest tests -v                       # Run tests
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Testing Strategy

### Test Coverage

**Backend Testing**:
- **Unit Tests**: Individual tool testing
- **Integration Tests**: End-to-end analysis testing
- **ML Model Tests**: Prediction accuracy validation
- **API Tests**: Endpoint functionality testing

**Frontend Testing**:
- **Component Tests**: React component testing
- **Integration Tests**: User flow testing
- **E2E Tests**: Complete application testing
- **Accessibility Tests**: WCAG compliance testing

### Testing Commands

```bash
# Backend tests
python -m pytest tests/ -v --cov=app --cov-report=xml

# Frontend tests
npm run test
npm run test:coverage
```

## 📈 Monitoring and Analytics

### Performance Monitoring

**Key Metrics**:
- **Response Times**: API endpoint performance
- **Error Rates**: System reliability tracking
- **Cache Hit Rates**: Caching effectiveness
- **ML Model Accuracy**: Prediction quality

**Logging Strategy**:
```python
import logging

# Structured logging for analysis
logger = logging.getLogger(__name__)

def log_analysis_result(ticker: str, timeframe: str, signal: str, confidence: float):
    logger.info({
        "ticker": ticker,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    })
```

## 🎯 Project Decisions and Rationale

### Technical Architecture Decisions

1. **LangChain for AI Orchestration**:
   - **Rationale**: Provides robust tool orchestration and memory management
   - **Benefits**: Handles complex multi-tool workflows, maintains context
   - **Trade-offs**: Additional complexity, but enables sophisticated AI behavior

2. **Technical-Only ML Features**:
   - **Rationale**: Fundamental data is difficult to obtain historically and align temporally
   - **Benefits**: Real-time performance, data consistency, proven effectiveness
   - **Trade-offs**: May miss some fundamental insights, but ensures reliability

3. **Multi-Source Data Pipeline**:
   - **Rationale**: Single data sources can fail or provide inconsistent data
   - **Benefits**: High reliability, data validation, automatic failover
   - **Trade-offs**: Increased complexity, but essential for production reliability

4. **In-Memory Caching**:
   - **Rationale**: Analysis computations are expensive and results are frequently reused
   - **Benefits**: Significant performance improvement, reduced API costs
   - **Trade-offs**: Memory usage, but critical for user experience

5. **FastAPI + Async/Await**:
   - **Rationale**: Modern Python web framework with excellent performance
   - **Benefits**: High performance, automatic documentation, type safety
   - **Trade-offs**: Learning curve, but provides excellent developer experience

### AI/ML Design Decisions

1. **Ensemble Model Approach**:
   - **Rationale**: Different models excel at different patterns
   - **Benefits**: Robust predictions, reduced overfitting, better generalization
   - **Trade-offs**: Increased complexity, but significantly better performance

2. **Low Temperature (0.1) for LLM**:
   - **Rationale**: Financial analysis requires consistency and reliability
   - **Benefits**: Deterministic results, consistent analysis quality
   - **Trade-offs**: Less creative responses, but essential for financial applications

3. **Multi-Timeframe Analysis**:
   - **Rationale**: Different investment strategies require different time horizons
   - **Benefits**: Comprehensive analysis, caters to different user needs
   - **Trade-offs**: Increased complexity, but provides complete investment picture

4. **Structured Output Format**:
   - **Rationale**: Consistent API responses enable reliable frontend integration
   - **Benefits**: Predictable responses, easier frontend development
   - **Trade-offs**: Less flexibility, but essential for production systems

## 🙏 Acknowledgments

- [Next.js](https://nextjs.org/) for the amazing React framework
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance Python API framework
- [Supabase](https://supabase.com/) for the backend-as-a-service platform
- [OpenAI](https://openai.com/) for the AI capabilities
- [LangChain](https://langchain.com/) for the AI application framework
- [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
- [TensorFlow](https://tensorflow.org/) for deep learning models
- [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) for data analysis