import uuid
import time
from typing import List, Optional
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.domain.entities.ocr_result import OCRResult
from backend.domain.entities.character import Character
from scripts.crnn_model import CRNN, CTCLabelConverter, create_crnn_model
from backend.application.services.ocr_service import OCRService


class OCRServiceWithCRNN(OCRService):
    """CRNN統合版OCRサービス"""
    
    def __init__(self):
        super().__init__()
        self.crnn_model = None
        self.crnn_converter = None
        self.device = torch.device('cpu')
        self._load_crnn_model()
    
    def _load_crnn_model(self):
        """CRNNモデルをロード"""
        try:
            model_path = project_root / 'models' / 'crnn_model_best.pth'
            
            if model_path.exists():
                self.crnn_model = create_crnn_model()
                checkpoint = torch.load(model_path, map_location=self.device)
                self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.crnn_model.to(self.device)
                self.crnn_model.eval()
                self.crnn_converter = CTCLabelConverter()
                print("CRNN model loaded successfully")
            else:
                print(f"CRNN model not found at {model_path}")
        except Exception as e:
            print(f"Error loading CRNN model: {e}")
            self.crnn_model = None
    
    def process_image(self, image_bytes: bytes) -> OCRResult:
        """画像を処理してOCR結果を返す（CRNN優先）"""
        start_time = time.time()
        image_id = f"img_{uuid.uuid4().hex[:8]}"
        
        try:
            # Convert bytes to numpy array
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # CRNNで全体を認識を試みる
            if self.crnn_model is not None:
                crnn_result = self._process_with_crnn(image_np)
                if crnn_result is not None:
                    # CRNNの結果を文字ごとのCharacterオブジェクトに変換
                    characters = []
                    for i, char in enumerate(crnn_result):
                        characters.append(Character(
                            granulate_symbol=f"G{char}",
                            latin_equivalent=char,
                            confidence=0.9  # CRNNは高い信頼度
                        ))
                    
                    processing_time = time.time() - start_time
                    return OCRResult(
                        image_id=image_id,
                        characters=characters,
                        processing_time=processing_time
                    )
            
            # CRNNが失敗した場合は従来の方法にフォールバック
            return super().process_image(image_bytes)
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return OCRResult(
                image_id=image_id,
                characters=[],
                processing_time=time.time() - start_time
            )
    
    def _process_with_crnn(self, image: np.ndarray) -> Optional[str]:
        """CRNNモデルで画像全体を認識"""
        if self.crnn_model is None:
            return None
        
        try:
            # 前処理
            preprocessed = self._preprocess_for_crnn(image)
            
            # テンソルに変換
            img_tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # 推論
            with torch.no_grad():
                output = self.crnn_model(img_tensor)
                _, preds = output.max(2)
                
                # デコード
                pred_text = self.crnn_converter.decode(preds)[0]
                
                # 空文字列の場合はNoneを返す
                if not pred_text:
                    return None
                
                return pred_text
                
        except Exception as e:
            print(f"CRNN processing error: {e}")
            return None
    
    def _preprocess_for_crnn(self, image: np.ndarray, target_height: int = 64, max_width: int = 256) -> np.ndarray:
        """CRNN用の前処理"""
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 背景色を判定
        mean_val = np.mean(gray)
        if mean_val > 128:
            # 白背景の場合は反転
            gray = 255 - gray
        
        # ノイズ除去
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 二値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # リサイズ（アスペクト比を保持）
        h, w = binary.shape
        aspect_ratio = w / h
        
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        
        # 最大幅を超える場合は調整
        if new_width > max_width:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        
        binary = cv2.resize(binary, (new_width, new_height))
        
        # パディング（左寄せ）
        padded = np.zeros((target_height, max_width), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        padded[y_offset:y_offset+new_height, :new_width] = binary
        
        # 正規化
        padded = padded.astype(np.float32) / 255.0
        
        return padded