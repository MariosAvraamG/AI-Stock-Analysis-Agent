from enum import Enum

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TimeFrame(str, Enum):
    SHORT = "short" #1-7 days
    MEDIUM = "medium" #1-4 weeks
    LONG = "long" #1-6 months

class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    ML_PREDICTION = "ml_prediction"
    MARKET_DATA = "market_data"