from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.ai_agent import stock_analysis_agent
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/ai-agent", tags=["AI Agent"])

class TradingSignalsRequest(BaseModel):
    ticker: str

@router.post("/signals")
async def get_trading_signals(request: TradingSignalsRequest, user_data: dict = Depends(get_current_user)):
    """
    Generate trading signals for short, medium, and long term timeframes.
    
    **Authentication Required**: Valid API key must be provided in the Authorization header.
    
    Returns structured signals with confidence levels for each timeframe:
    - **SHORT TERM (5-20 days)**: Based on technical momentum and short-term sentiment
    - **MEDIUM TERM (20-60 days)**: Based on trend analysis and market sentiment  
    - **LONG TERM (60+ days)**: Based on fundamental analysis and ML predictions
    
    Each signal includes:
    - signal: BUY, SELL, or HOLD
    - confidence: 0.0 to 1.0 (higher = more confident)
    - reasoning: Brief explanation of the signal
    
    - **ticker**: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
    """
    try:
        if not request.ticker or len(request.ticker.strip()) == 0:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        
        result = stock_analysis_agent.get_trading_signals(ticker=request.ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals/demo")
async def get_trading_signals_demo(request: TradingSignalsRequest):
    """
    Demo trading signals for short, medium, and long term timeframes.
    """
    try:
        if not request.ticker or len(request.ticker.strip()) == 0:
            raise HTTPException(status_code=400, detail="Ticker symbol is required")
        
        result = stock_analysis_agent.get_trading_signals(ticker=request.ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))