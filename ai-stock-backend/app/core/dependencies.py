from fastapi import Header, HTTPException, Depends
from typing import Dict, Any
from app.core.auth import validate_api_key

async def get_api_key_from_header(
    authorization: str = Header(..., description="Bearer <api_key>")
) -> str:
    """
    Extract API key from Authorization header
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Invalid Authorization header format. Use: Bearer <api_key>"
        )
    
    api_key = authorization[7:] #Remove "Bearer " prefix

    if not api_key.strip():
        raise HTTPException(status_code=401, detail="API key cannot be empty")
    
    return api_key.strip()

async def get_current_user(
    api_key: str = Depends(get_api_key_from_header)
) -> Dict[str, Any]:
    """
    Validate API key and return user data
    """
    return await validate_api_key(api_key)

def require_permission(permission: str):
    """Dependency factory to require specific permissions"""
    async def permission_checker(user_data: Dict[str, Any] = Depends(get_current_user)):
        permissions = user_data.get("permissions", {})
        if not permissions.get(permission, False):
            raise HTTPException(
                status_code=403, 
                detail=f"This operation requires '{permission}' permission"
            )
        return user_data
    return permission_checker