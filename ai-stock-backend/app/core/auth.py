import hashlib
import httpx
from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException
from app.core.config import settings

class APIKeyValidator:
    def __init__(self):
        self.base_url = settings.supabase_url.rstrip("/")
        self.service_key = settings.supabase_service_key
        self.headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
    
    async def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key using direct HTTP requests to Supabase REST API
        """

        #Step 1: Validate API key format
        if not api_key or not isinstance(api_key, str):
            raise HTTPException(status_code=401, detail="API key is required")
        
        if not api_key.startswith("ak_"):
            raise HTTPException(status_code=401, detail="Invalid API key format")

        if len(api_key) < 10: #Minimum reasonable length
            raise HTTPException(status_code=401, detail="Invalid API key format")

        #Step 2: Hash the API key (Same method as Next.js)
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid API key format")
        
        #Step 3: Query Supabase database via REST API
        try:
            url = f"{self.base_url}/rest/v1/api_keys"
            params = {
                "key_hash": f"eq.{key_hash}",
                "is_active": "eq.true",
                "select": "user_id, permissions, is_active, expires_at, last_used_at, name"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, headers=self.headers)

                if response.status_code == 401:
                    raise HTTPException(status_code=500, detail="Supabase authentication failed - check service key")
                elif response.status_code == 404:
                    raise HTTPException(status_code=500, detail="API keys table not found")
                elif response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Supabase error: {response.status_code}")
                
                data = response.json()

                if not data or len(data) == 0:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                api_key_data = data[0]
            
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Authentication service error: {str(e)}")
        except HTTPException as e:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")


        #Step 4: Check expiration
        if api_key_data.get('expires_at'):
            try:
                expires_at_str = api_key_data['expires_at']

                #Handle different datetime formats from Supabase
                if expires_at_str.endswith("Z"):
                    expires_at_str = expires_at_str[:-1] + "+00:00"

                expires_at = datetime.fromisoformat(expires_at_str)

                if datetime.now().replace(tzinfo=expires_at.tzinfo) > expires_at:
                    try:
                        update_url = f"{self.base_url}/rest/v1/api_keys"
                        update_params = {
                            "key_hash": f"eq.{key_hash}"
                        }
                        update_data = {
                            "is_active": "eq.false"
                        }
                    
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await client.patch(
                                update_url, 
                                params=update_params, 
                                json=update_data, 
                                headers=self.headers
                            )
                    except Exception as e:
                        #Don't fail the request if we can't update is_active
                        print(f"Warning: Could not update is_active: {str(e)}")
                    raise HTTPException(status_code=401, detail="API key has expired")
            except(ValueError, TypeError):
                #If we can't parse the expiration date, log it but don't fail
                print(f"Warning: Could not parse expiration date: {api_key_data.get('expires_at')}")

        #Step 5: Update last_used_at
        try:
            update_url = f"{self.base_url}/rest/v1/api_keys"
            update_params = {
                "key_hash": f"eq.{key_hash}"
            }
            update_data = {
                "last_used_at": datetime.now().isoformat() + "Z"
            }

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.patch(
                    update_url, 
                    params=update_params, 
                    json=update_data, 
                    headers=self.headers
                )
        
        except Exception as e:
            #Don't fail the request if we can't update last_used_at
            print(f"Warning: Could not update last_used_at: {str(e)}")

        #Step 6: Return validated user data
        return{
            "user_id": api_key_data["user_id"],
            "permissions": api_key_data.get("permissions", {}),
            "key_name": api_key_data.get("name", "Unknown"),
            "api_key_hash": key_hash
        }

#Create global instance
api_key_validator = APIKeyValidator()

#Convenience function for FastAPI dependency injection
async def validate_api_key(api_key: str) -> Dict[str, Any]:
    return await api_key_validator.validate_api_key(api_key)