from pydantic import BaseModel
from typing import Dict, Any, Optional

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_seconds: float