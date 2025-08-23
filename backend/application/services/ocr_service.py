import uuid
import time
from typing import List, Tuple, Optional
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import pytesseract
from backend.domain.entities.ocr_result import OCRResult
from backend.domain.entities.character import Character
from backend.infrastructure.mapping.granulate_alphabet_generated import GranulateAlphabet


class OCRService:
    def __init__(self):
        self.alphabet = GranulateAlphabet()

    def process_image(self, image_bytes: bytes) -> OCRResult:
        start_time = time.time()
        image_id = f"img_{uuid.uuid4().hex[:8]}"

        try:
            # Convert bytes to numpy array
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Preprocess image
            preprocessed = self._preprocess_image(image_np)
            
            # Extract character regions
            char_regions = self._extract_character_regions(preprocessed)
            
            # Recognize each character
            characters: List[Character] = []
            for i, (x, y, w, h) in enumerate(char_regions):
                char_image = preprocessed[y:y+h, x:x+w]
                
                # まずTesseractで試す
                recognized_char = self.process_with_tesseract(char_image)
                
                # Tesseractで認識できない場合はHash-basedにフォールバック
                if not recognized_char:
                    recognized_char = self.alphabet.compare_image_to_mapping(char_image)
                
                if recognized_char:
                    # Simple confidence based on image quality
                    confidence = 0.8 if w > 20 and h > 20 else 0.6
                    
                    characters.append(Character(
                        granulate_symbol=f"G{recognized_char}",  # Placeholder granulate representation
                        latin_equivalent=recognized_char,
                        confidence=confidence,
                        position={'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                    ))

            processing_time = time.time() - start_time

            return OCRResult(
                image_id=image_id, 
                characters=characters, 
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            # Return empty result on error
            return OCRResult(
                image_id=image_id, 
                characters=[], 
                processing_time=time.time() - start_time
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply bilateral filter for noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply binary threshold (aggressive mode from frontend)
        _, binary = cv2.threshold(enhanced, 128, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def process_with_tesseract(self, char_image: np.ndarray) -> Optional[str]:
        """Tesseractカスタムモデルでの文字認識"""
        try:
            import pytesseract
            # カスタム言語（gran）を使用
            text = pytesseract.image_to_string(
                char_image,
                lang='gran',
                config='--oem 0 --psm 10'  # レガシーエンジン、単一文字モード
            )
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Tesseract error: {e}")
            return None
    
    def _extract_character_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract character regions from preprocessed image"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours
        char_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size - adjust these thresholds based on your images
            if w > 10 and h > 10 and w < image.shape[1] * 0.5 and h < image.shape[0] * 0.5:
                char_regions.append((x, y, w, h))
        
        # Sort by x coordinate (left to right)
        char_regions.sort(key=lambda r: r[0])
        
        return char_regions
