from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
import uvicorn

from app.core.config import settings
from app.core.dependencies import require_permission, get_current_user
from app.services.tools.multi_source_data import multi_source_tool
from app.api.ai_agent import router as ai_agent_router

app = FastAPI(
    title="AI Stock Analysis API",
    description="AI-powered stock analysis API with LangChain agent",
    version="1.0.0"
)

#Include AI agent router
app.include_router(ai_agent_router)

#CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AI Stock Analysis API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }

#Test authentication endpoint
@app.get("/auth/test")
async def test_auth(user_data: dict = Depends(get_current_user)):
    """Test API key authentication"""
    return {
        "message": "Authentication successful!",
        "user_id": user_data["user_id"],
        "permissions": user_data["permissions"],
        "key_name": user_data["key_name"]
    }

#Test read permission-based endpoint
@app.get("/auth/test-read")
async def test_read_permission(user_data: dict = Depends(require_permission("read"))):
    """Test read permission requirement"""
    return {
        "message": "Read permission verified!",
        "user_id": user_data["user_id"],
        "data": "This is protected data for read permission"
    }

#Test write permission-based endpoint for future expansions
@app.get("/auth/test-write")
async def test_write_permission(user_data: dict = Depends(require_permission("write"))):
    """Test write permission requirement"""
    return {
        "message": "Write permission verified!",
        "user_id": user_data["user_id"],
        "data": "This is protected data for write permission"
    }

#Test market data tool
@app.get("/market-data/{ticker}")
async def get_market_data(ticker: str):
    """Test market data tool"""
    result = multi_source_tool.get_stock_data(ticker)
    return {
        "tool_result": result.dict(),
        "summary": {
            "success": result.success,
            "ticker": ticker.upper(),
            "execution_time": f"{result.execution_time_seconds:.2f}s",
            "current_price": result.data.get("current_price") if result.success else None,
            "company": result.data.get("company_name") if result.success else None
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.debug else False
    )