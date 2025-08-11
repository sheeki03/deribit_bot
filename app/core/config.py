from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # Database (optional - system works without it)
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    # Firecrawl (no API key needed)
    firecrawl_api_key: Optional[str] = Field(None, env="FIRECRAWL_API_KEY") 
    firecrawl_base_url: str = Field(..., env="FIRECRAWL_BASE_URL")
    firecrawl_api_url: str = Field(..., env="FIRECRAWL_API_URL")
    
    # AI/ML (OpenRouter)
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    
    # Telegram (optional for now)
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    
    # Market Data (CoinGecko)
    coingecko_api_key: Optional[str] = Field(None, env="COINGECKO_API_KEY")
    coingecko_base_url: str = Field("https://api.coingecko.com/api/v3", env="COINGECKO_BASE_URL")
    
    # Redis (optional - system works without it)
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    
    # Application
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    webhook_secret: Optional[str] = Field(None, env="WEBHOOK_SECRET")
    
    # Scraping
    user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        env="USER_AGENT"
    )
    request_delay: float = Field(1.0, env="REQUEST_DELAY")
    max_retries: int = Field(3, env="MAX_RETRIES")
    
    # Paths
    data_dir: str = Field("./data", env="DATA_DIR")
    images_dir: str = Field("./data/images", env="IMAGES_DIR")
    logs_dir: str = Field("./logs", env="LOGS_DIR")
    
    # Streamlit
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Scoring thresholds
    min_confidence_threshold: float = Field(0.7, env="MIN_CONFIDENCE_THRESHOLD")
    alert_threshold: float = Field(0.3, env="ALERT_THRESHOLD")
    extreme_threshold: float = Field(0.5, env="EXTREME_THRESHOLD")
    
    # Processing limits
    max_images_per_article: int = Field(20, env="MAX_IMAGES_PER_ARTICLE")
    image_timeout_seconds: int = Field(30, env="IMAGE_TIMEOUT_SECONDS")
    ocr_max_retries: int = Field(2, env="OCR_MAX_RETRIES")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


# Global settings instance
settings = Settings()