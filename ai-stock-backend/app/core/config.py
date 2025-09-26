from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
class Settings(BaseSettings):
    #Supabase
    supabase_url: Optional[str] = None
    supabase_service_key: Optional[str] = None

    #OpenAI
    openai_api_key: Optional[str] = None

    #Alpha Vantage
    alpha_vantage_api_key: Optional[str] = None

    #NewsAPI
    news_api_key: Optional[str] = None
    
    #Twitter API
    twitter_bearer_token: Optional[str] = None

    #App settings
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")

    def validate_supabase_config(self):
        """Validate Supabase configuration"""
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not self.supabase_service_key:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable is required")
        
        if not self.supabase_url.startswith('https://'):
            raise ValueError("SUPABASE_URL must start with https://")

settings = Settings()

# Validate Supabase config
try:
    settings.validate_supabase_config()
    print("✅ Supabase configuration validated")
except ValueError as e:
    print(f"❌ Supabase configuration error: {e}")