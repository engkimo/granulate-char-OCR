import uuid
import time
from typing import List, Tuple, Optional
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import pytesseract
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from backend.domain.entities.ocr_result import OCRResult
from backend.domain.entities.character import Character
from backend.infrastructure.mapping.granulate_alphabet_generated import GranulateAlphabet


class GranulateOCRModel(nn.Module):
    """グラニュート文字認識用のCNNモデル"""
    
    def __init__(self, num_classes=26):
        super().__init__()
        
        # 特徴抽出層
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class OCRService:
    def __init__(self):
        self.alphabet = GranulateAlphabet()
        self.cnn_model = None
        self.device = torch.device('cpu')
        self._load_cnn_model()

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
                
                # 1. まずCNNモデルで試す（最高精度）
                recognized_char, cnn_confidence = self.process_with_cnn(char_image)
                
                # 2. CNNで認識できない場合はTesseractを試す
                # 闾値を下げてCNNを優先
                if not recognized_char or cnn_confidence < 0.3:
                    tesseract_char = self.process_with_tesseract(char_image)
                    if tesseract_char:
                        recognized_char = tesseract_char
                        confidence = 0.7  # Tesseractの信頼度
                    else:
                        confidence = cnn_confidence if recognized_char else 0.0
                else:
                    confidence = cnn_confidence
                
                # 3. それでも認識できない場合はHash-basedにフォールバック
                if not recognized_char:
                    recognized_char = self.alphabet.compare_image_to_mapping(char_image)
                    confidence = 0.5 if recognized_char else 0.0
                
                if recognized_char:
                    
                    characters.append(Character(
                        granulate_symbol=f"G{recognized_char}",  # Placeholder granulate representation
                        latin_equivalent=recognized_char,
                        confidence=confidence
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
            
        # 画像の背景色を判定（平均値が128以上なら白背景）
        mean_val = np.mean(gray)
        if mean_val > 128:
            # 白背景の場合は反転（黒背景白文字に変換）
            gray = 255 - gray
            
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
            # 元画像が64x64で、文字が約45x47なので、閾値を調整
            if w > 5 and h > 5:  # 最小サイズを緩和
                char_regions.append((x, y, w, h))
        
        # Sort by x coordinate (left to right)
        char_regions.sort(key=lambda r: r[0])
        
        return char_regions
    
    def _load_cnn_model(self):
        """CNNモデルをロード"""
        try:
            model_path = Path('models/cnn_model_best.pth')
            if model_path.exists():
                self.cnn_model = GranulateOCRModel(num_classes=26)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.cnn_model.to(self.device)
                self.cnn_model.eval()
                print("CNN model loaded successfully")
                
                # 変換処理の準備
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                
                # インデックスから文字へのマッピング
                self.idx_to_class = {i: chr(ord('A') + i) for i in range(26)}
            else:
                print(f"CNN model not found at {model_path}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            self.cnn_model = None
    
    def process_with_cnn(self, char_image: np.ndarray) -> Tuple[Optional[str], float]:
        """CNNモデルでの文字認識"""
        if self.cnn_model is None:
            return None, 0.0
        
        try:
            # 画像の前処理
            # CNNは白背景黒文字で学習されている
            # 現在の画像が黒背景白文字の場合は反転
            # 白いピクセルの割合を確認
            white_pixels = np.sum(char_image > 128)
            total_pixels = char_image.size
            white_ratio = white_pixels / total_pixels
            
            # 白いピクセルが50%以上なら背景が白、そうでなければ文字が白
            if white_ratio < 0.5:  # 黒背景白文字の場合
                char_image = 255 - char_image
            
            # テンソルに変換
            img_tensor = self.transform(char_image).unsqueeze(0).to(self.device)
            
            # 推論
            with torch.no_grad():
                outputs = self.cnn_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                char_idx = predicted.item()
                confidence_value = confidence.item()
                
                if char_idx in self.idx_to_class:
                    return self.idx_to_class[char_idx], confidence_value
                else:
                    return None, 0.0
                    
        except Exception as e:
            print(f"CNN processing error: {e}")
            return None, 0.0
