"""Image processing and augmentation utilities for OCR."""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Union, Tuple, List, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


class ImageProcessor:
    """Advanced image processing for OCR optimization."""
    
    @staticmethod
    def compute_hash(image_path: Union[str, Path]) -> str:
        """Compute MD5 hash of image file."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    @staticmethod
    def load_image(image_path: Union[str, Path, bytes]) -> Image.Image:
        """Load image from file or bytes."""
        if isinstance(image_path, bytes):
            return Image.open(BytesIO(image_path))
        return Image.open(image_path)
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: Tuple[int, int] = (2000, 2000)) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    
    @staticmethod
    def convert_to_grayscale(image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        return image.convert('L')
    
    @staticmethod
    def enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
        """Enhance image contrast."""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def enhance_sharpness(image: Image.Image, factor: float = 2.0) -> Image.Image:
        """Enhance image sharpness."""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def remove_noise(image: Image.Image) -> Image.Image:
        """Remove noise using median filter."""
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    @staticmethod
    def binarize(image: Image.Image, threshold: int = 128) -> Image.Image:
        """Convert to binary image."""
        if image.mode != 'L':
            image = image.convert('L')
        return image.point(lambda x: 255 if x > threshold else 0, mode='1')
    
    @staticmethod
    def auto_rotate(image: Image.Image) -> Image.Image:
        """Auto-rotate image based on EXIF orientation."""
        try:
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274 is the orientation tag
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except:
            pass
        return image
    
    @staticmethod
    def preprocess_for_ocr(image: Image.Image, aggressive: bool = False) -> Image.Image:
        """Complete preprocessing pipeline for OCR."""
        # Auto-rotate
        image = ImageProcessor.auto_rotate(image)
        
        # Resize if too large
        image = ImageProcessor.resize_image(image)
        
        # Convert to grayscale
        image = ImageProcessor.convert_to_grayscale(image)
        
        if aggressive:
            # Remove noise
            image = ImageProcessor.remove_noise(image)
            
            # Enhance contrast
            image = ImageProcessor.enhance_contrast(image, factor=2.0)
            
            # Enhance sharpness
            image = ImageProcessor.enhance_sharpness(image, factor=2.5)
        else:
            # Mild enhancement
            image = ImageProcessor.enhance_contrast(image, factor=1.5)
            image = ImageProcessor.enhance_sharpness(image, factor=1.5)
        
        return image
    
    @staticmethod
    def assess_quality(image: Image.Image) -> Dict[str, Any]:
        """Assess image quality."""
        # Convert to numpy array
        img_array = np.array(image.convert('L'))
        
        # Calculate metrics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Simple sharpness metric using variance
        # In production, consider using Laplacian-based methods for better accuracy
        image_variance = np.var(img_array)
        
        # Determine quality
        quality = "good"
        if mean_brightness < 50 or mean_brightness > 200:
            quality = "poor_lighting"
        elif std_brightness < 30:
            quality = "low_contrast"
        elif image_variance < 100:
            quality = "blurry"
        
        return {
            'quality': quality,
            'brightness': float(mean_brightness),
            'contrast': float(std_brightness),
            'sharpness': float(image_variance)
        }
    
    @staticmethod
    def pdf_to_images(pdf_path: Union[str, Path], dpi: int = 300) -> List[Image.Image]:
        """Convert PDF to images."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except ImportError:
            print("pdf2image not installed. Install with: pip install pdf2image")
            return []
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []
