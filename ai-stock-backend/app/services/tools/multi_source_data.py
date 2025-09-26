import time
from typing import Dict, Any, List
from app.models.schemas import ToolResult
from app.services.tools.yahoo_finance_data import market_data_tool
from app.services.tools.alpha_vantage_data import alpha_vantage_tool

class MultiSourceDataTool:
    """Multi-source data tool with fallback options"""
    
    def __init__(self):
        self.name = "multi_source_data"
        self.sources = [
            ("yfinance", market_data_tool),
            ("alpha_vantage", alpha_vantage_tool)
        ]
    
    def get_stock_data(self, ticker: str) -> ToolResult:
        """Try multiple data sources until one succeeds"""
        start_time = time.time()
        errors = []
        
        for source_name, source_tool in self.sources:
            try:
                print(f"🔄 Trying {source_name} for {ticker}")
                result = source_tool.get_stock_data(ticker)
                
                if result.success:
                    print(f"✅ {source_name} succeeded for {ticker}")
                    # Add source info to result
                    result.data["primary_source"] = source_name
                    return result
                else:
                    errors.append(f"{source_name}: {result.error_message}")
                    print(f"❌ {source_name} failed: {result.error_message}")
            
            except Exception as e:
                error_msg = f"{source_name}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {source_name} exception: {str(e)}")
        
        # All sources failed
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={},
            error_message=f"All data sources failed: {'; '.join(errors)}",
            execution_time_seconds=time.time() - start_time
        )
    
    def get_historical_data(self, ticker: str, period: str = "1y") -> ToolResult:
        """Get historical OHLCV data with multi-source fallback"""
        start_time = time.time()
        errors = []
        
        for source_name, source_tool in self.sources:
            try:
                print(f"🔄 Trying {source_name} for historical data of {ticker}")
                result = source_tool.get_historical_data(ticker, period)
                
                if result.success:
                    print(f"✅ {source_name} succeeded for historical data of {ticker}")
                    # Add source info to result
                    result.data["primary_source"] = source_name
                    return result
                else:
                    errors.append(f"{source_name}: {result.error_message}")
                    print(f"❌ {source_name} failed for historical data: {result.error_message}")
            
            except Exception as e:
                error_msg = f"{source_name}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {source_name} exception for historical data: {str(e)}")
        
        # All sources failed
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={},
            error_message=f"All historical data sources failed: {'; '.join(errors)}",
            execution_time_seconds=time.time() - start_time
        )
    
    def get_fundamental_data(self, ticker: str) -> ToolResult:
        """Get comprehensive fundamental data with multi-source fallback"""
        start_time = time.time()
        errors = []
        
        for source_name, source_tool in self.sources:
            try:
                print(f"🔄 Trying {source_name} for fundamental data of {ticker}")
                result = source_tool.get_fundamental_data(ticker)
                
                if result.success:
                    print(f"✅ {source_name} succeeded for fundamental data of {ticker}")
                    # Add source info to result
                    result.data["primary_source"] = source_name
                    return result
                else:
                    errors.append(f"{source_name}: {result.error_message}")
                    print(f"❌ {source_name} failed for fundamental data: {result.error_message}")
            
            except Exception as e:
                error_msg = f"{source_name}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {source_name} exception for fundamental data: {str(e)}")
        
        # All sources failed
        return ToolResult(
            tool_name=self.name,
            success=False,
            data={},
            error_message=f"All fundamental data sources failed: {'; '.join(errors)}",
            execution_time_seconds=time.time() - start_time
        )

# Global instance
multi_source_tool = MultiSourceDataTool()