
# Optimized preprocessing for Granulate OCR
import cv2
import numpy as np

class GranulatePreprocessor:
    """Optimized preprocessing for Granulate character recognition"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    def process(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # Enhance contrast
        enhanced = self.clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 3
        )
        
        # Clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return final
