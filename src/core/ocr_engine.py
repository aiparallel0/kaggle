"""OCR engines for text extraction from receipt images."""

import logging
from typing import List, Dict
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine:
    """Base OCR engine interface."""
    
    def extract_text(self, image: Image.Image) -> List[Dict]:
        """Extract text from image. Returns list of text boxes with coordinates."""
        raise NotImplementedError


class EasyOCREngine(OCREngine):
    """EasyOCR implementation.
    
    Args:
        languages: List of language codes to use for OCR. Defaults to Config.OCR_LANGUAGES.
        gpu: Whether to use GPU acceleration. Defaults to False.
        confidence_threshold: Minimum confidence score (0.0-1.0) to accept text. Defaults to 0.3.
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False, confidence_threshold: float = 0.3):
        from ..utils.config import Config
        self.languages = languages or Config.OCR_LANGUAGES
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.reader = None
        self._initialize()
    
    def _initialize(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages: {self.languages[:5]}...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info("✅ EasyOCR initialized successfully")
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
            raise
    
    def extract_text(self, image: Image.Image) -> List[Dict]:
        """Extract text using EasyOCR."""
        if self.reader is None:
            self._initialize()
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = self.reader.readtext(img_array)
        
        # Convert to standard format
        text_boxes = []
        for bbox, text, confidence in results:
            if confidence >= self.confidence_threshold:
                text_boxes.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'position': {
                        'x': int(bbox[0][0]),
                        'y': int(bbox[0][1]),
                        'width': int(bbox[2][0] - bbox[0][0]),
                        'height': int(bbox[2][1] - bbox[0][1])
                    }
                })
        
        return text_boxes


class TesseractEngine(OCREngine):
    """Tesseract OCR implementation."""
    
    def __init__(self, languages: str = 'eng', confidence_threshold: float = 0.3):
        self.languages = languages
        self.confidence_threshold = confidence_threshold
    
    def extract_text(self, image: Image.Image) -> List[Dict]:
        """Extract text using Tesseract."""
        try:
            import pytesseract
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                output_type=pytesseract.Output.DICT
            )
            
            # Convert to standard format
            text_boxes = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf >= self.confidence_threshold * 100:
                    text_boxes.append({
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': None,
                        'position': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            return text_boxes
        except ImportError:
            logger.error("Pytesseract not installed. Install with: pip install pytesseract")
            return []
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return []


class HybridOCREngine(OCREngine):
    """Hybrid OCR engine using multiple engines."""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.engines = {}
        self.primary_engine = 'easyocr'
        self.fallback_engine = 'tesseract'
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all OCR engines."""
        try:
            self.engines['easyocr'] = EasyOCREngine(confidence_threshold=self.confidence_threshold)
        except Exception as e:
            logger.warning(f"Could not initialize EasyOCR: {e}")
        
        try:
            self.engines['tesseract'] = TesseractEngine(confidence_threshold=self.confidence_threshold)
        except Exception as e:
            logger.warning(f"Could not initialize Tesseract: {e}")
    
    def extract_text(self, image: Image.Image) -> List[Dict]:
        """Extract text using primary engine with fallback."""
        if self.primary_engine in self.engines:
            try:
                results = self.engines[self.primary_engine].extract_text(image)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Primary engine failed: {e}")
        
        if self.fallback_engine in self.engines:
            try:
                logger.info("Using fallback OCR engine...")
                return self.engines[self.fallback_engine].extract_text(image)
            except Exception as e:
                logger.error(f"Fallback engine also failed: {e}")
        
        logger.error("No OCR engines available")
        return []
