"""Configuration management for OCR receipt analysis system."""

import logging
from pathlib import Path
from datetime import datetime


class Config:
    """Global configuration for the OCR receipt analysis system."""
    
    # Application settings
    APP_NAME = "OCR Receipt Analysis Mega System"
    VERSION = "2.0.0"
    DEBUG = False
    
    # Directories
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / "cache"
    TEMP_DIR = BASE_DIR / "temp"
    
    # OCR settings
    OCR_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko', 'ar', 
                     'ru', 'hi', 'th', 'vi', 'id', 'nl', 'pl', 'tr', 'sv', 'da']
    OCR_GPU = False
    OCR_CONFIDENCE_THRESHOLD = 0.3
    
    # Image processing
    IMAGE_MAX_SIZE = (2000, 2000)
    IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    PDF_DPI = 300
    
    # ML settings
    CLASSIFIER_ALGORITHM = 'random_forest'  # Options: naive_bayes, random_forest, svm, logistic
    TRAIN_TEST_SPLIT = 0.2
    CV_FOLDS = 5
    GRID_SEARCH = False
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4
    ENABLE_CORS = True
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_PERIOD = 60  # seconds
    
    # Database
    DB_TYPE = 'sqlite'  # Options: sqlite, postgresql, mongodb
    SQLITE_DB = DATA_DIR / "receipts.db"
    
    # Caching
    ENABLE_CACHE = True
    CACHE_TTL = 3600  # seconds
    
    # Processing
    MAX_WORKERS = 4
    BATCH_SIZE = 10
    TIMEOUT = 300  # seconds
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / f"receipt_ocr_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Known stores for better recognition
    KNOWN_STORES = [
        'WALMART', 'TARGET', 'COSTCO', 'KROGER', 'SAFEWAY', 'WHOLE FOODS',
        'TRADER JOES', 'ALDI', 'PUBLIX', 'WEGMANS', 'MEIJER', 'HEB',
        'CVSPHAR MACY', 'WALGREENS', 'RITE AID', 'STARBUCKS', 'MCDONALDS',
        'SUBWAY', 'CHIPOTLE', 'PANERA', 'OLIVE GARDEN', 'APPLEBEES',
        'AMAZON', 'EBAY', 'BEST BUY', 'HOME DEPOT', 'LOWES', 'IKEA',
        'MOMI', 'RALPHS', 'VONS', 'ALBERTSONS', 'FOOD LION', 'GIANT',
        'STOP SHOP', 'SHOPRITE', 'HARRIS TEETER', 'FRED MEYER',
        'SEVEN ELEVEN', '7-ELEVEN', 'CIRCLE K', 'SPEEDWAY', 'SHELL',
        'BP', 'CHEVRON', 'EXXON', 'MOBIL', 'TEXACO', 'ARCO',
        'UBER EATS', 'DOORDASH', 'GRUBHUB', 'POSTMATES', 'INSTACART'
    ]
    
    # Currency patterns
    CURRENCIES = {
        'USD': r'\$',
        'EUR': r'€',
        'GBP': r'£',
        'JPY': r'¥',
        'CNY': r'¥|元',
        'INR': r'₹',
        'RUB': r'₽',
        'BRL': r'R\$',
        'MXN': r'MXN|\$',
        'CAD': r'CAD|\$'
    }
    
    # Date formats
    DATE_FORMATS = [
        r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
        r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
        r'\d{2}-\d{2}-\d{4}',  # 01-15-2024
        r'\d{2}\.\d{2}\.\d{4}',  # 01.15.2024
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # 1/15/24
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # 1-15-24
        r'\b\d{8}\b',  # 20240115
        r'\b[A-Za-z]{3}\s+\d{1,2},?\s+\d{4}\b',  # Jan 15, 2024
        r'\b\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\b',  # 15 Jan 2024
    ]
    
    # Time formats
    TIME_FORMATS = [
        r'\d{2}:\d{2}:\d{2}',  # 14:30:25
        r'\d{2}:\d{2}',  # 14:30
        r'\d{1,2}:\d{2}\s*[AP]M',  # 2:30 PM
    ]
    
    # Tax patterns
    TAX_PATTERNS = [
        r'(?i)tax\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)vat\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)gst\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)sales\s*tax\s*:?\s*\$?(\d+\.?\d*)',
    ]
    
    # Total patterns
    TOTAL_PATTERNS = [
        r'(?i)total\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)amount\s*due\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)balance\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)grand\s*total\s*:?\s*\$?(\d+\.?\d*)',
        r'(?i)sum\s*:?\s*\$?(\d+\.?\d*)',
    ]
    
    # Payment methods
    PAYMENT_METHODS = [
        'CASH', 'CREDIT', 'DEBIT', 'VISA', 'MASTERCARD', 'AMEX', 
        'AMERICAN EXPRESS', 'DISCOVER', 'PAYPAL', 'VENMO', 'APPLE PAY',
        'GOOGLE PAY', 'SAMSUNG PAY', 'CHECK', 'GIFT CARD', 'EBT'
    ]
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.MODELS_DIR, 
                         cls.LOGS_DIR, cls.CACHE_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_classifier_path(cls):
        """Get path for classifier model."""
        return cls.MODELS_DIR / f"receipt_classifier_{cls.CLASSIFIER_ALGORITHM}.pkl"
    
    @classmethod
    def get_vectorizer_path(cls):
        """Get path for text vectorizer."""
        return cls.MODELS_DIR / "text_vectorizer.pkl"
