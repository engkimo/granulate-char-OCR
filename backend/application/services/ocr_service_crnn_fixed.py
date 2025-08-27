import time
import uuid
from io import BytesIO
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image
import torch

from backend.application.services.ocr_service import OCRService
from backend.domain.entities.character import Character
from backend.domain.entities.ocr_result import OCRResult
from scripts.crnn_model import CRNN, CTCLabelConverter
from pathlib import Path
import json


class OCRServiceWithCRNNFixed(OCRService):
    """CRNN統合版OCRサービス（修正版）"""
    
    def __init__(self):
        # CRNN専用のためCNN読み込みを抑制
        super().__init__(load_cnn=False)
        self.crnn_model = None
        self.crnn_converter = None
        self._load_crnn_model()
        print("Initialized OCRServiceWithCRNNFixed (CNN disabled, CRNN primary)")
        
        # グラニュート文字マッピングをロード
        self.latin_to_granulate = self._load_character_mapping()
    
    def _load_character_mapping(self):
        """グラニュート文字マッピングをロード"""
        mapping = {}
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            mapping_path = project_root / 'backend' / 'infrastructure' / 'mapping' / 'granulate_character_data.json'
            
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSONのキーがラテン文字なので、そのまま使用
                    for latin_char in data.keys():
                        # 実際のグラニュート文字の表示方法は要確認
                        # ここでは仮に "G" + ラテン文字 で表現
                        mapping[latin_char] = f"◆{latin_char}"
                print(f"Character mapping loaded: {len(mapping)} characters")
            else:
                print(f"Character mapping file not found at {mapping_path}")
        except Exception as e:
            print(f"Failed to load character mapping: {e}")
        
        return mapping
    
    def _load_crnn_model(self):
        """CRNNモデルをロード"""
        try:
            project_root = Path(__file__).parent.parent.parent.parent
            model_path = project_root / 'models' / 'crnn_model_best.pth'
            
            if model_path.exists():
                self.crnn_model = CRNN(
                    img_height=64,
                    num_classes=27,  # A-Z + blank
                    hidden_size=256
                )
                checkpoint = torch.load(str(model_path), map_location=self.device)
                self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.crnn_model.to(self.device)
                self.crnn_model.eval()
                self.crnn_converter = CTCLabelConverter()
                print("CRNN model loaded successfully")
            else:
                print(f"CRNN model not found at {model_path}")
        except Exception as e:
            print(f"Failed to load CRNN model: {e}")
    
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
                if crnn_result is not None and crnn_result['text']:
                    # CRNNの結果を文字ごとのCharacterオブジェクトに変換
                    characters = []
                    for i, char in enumerate(crnn_result['text']):
                        # ラテン文字から対応するグラニュート文字を取得
                        granulate_symbol = self.latin_to_granulate.get(char, f"?{char}")
                        
                        characters.append(Character(
                            granulate_symbol=granulate_symbol,
                            latin_equivalent=char,
                            confidence=crnn_result['confidence']
                        ))
                    
                    processing_time = time.time() - start_time
                    return OCRResult(
                        image_id=image_id,
                        characters=characters,
                        processing_time=processing_time
                    )
            
            # CRNNが失敗した場合は従来パイプライン（セグメンテーション + 可能ならCNN/Tesseract）にフォールバック
            print("CRNN failed, falling back to classic pipeline")
            return super().process_image(image_bytes)
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            import traceback
            traceback.print_exc()
            return OCRResult(
                image_id=image_id,
                characters=[],
                processing_time=time.time() - start_time
            )
    
    def _process_with_crnn(self, image: np.ndarray) -> Optional[dict]:
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
                
                # ソフトマックスを適用して確率を取得
                probs = torch.softmax(output, dim=2)
                
                # 最大確率のインデックスを取得
                max_probs, preds = probs.max(2)
                
                # デコード
                pred_text = self.crnn_converter.decode(preds)[0]
                
                # 平均信頼度を計算（blank以外の文字の確率の平均）
                confidence_scores = []
                for i, idx in enumerate(preds[0]):
                    if idx != 0:  # blank以外
                        confidence_scores.append(max_probs[0][i].item())
                
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
                
                # 空文字列の場合はNoneを返す
                if not pred_text:
                    return None
                
                return {
                    'text': pred_text,
                    'confidence': avg_confidence
                }
                
        except Exception as e:
            print(f"CRNN processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_for_crnn(self, image: np.ndarray, target_height: int = 64, max_width: int = 256) -> np.ndarray:
        """CRNN用の前処理（より穏やかな処理）
        入力高さは常に64に固定する（モデルと整合）
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 背景色を判定
        mean_val = np.mean(gray)
        if mean_val > 128:
            # 白背景の場合は反転
            gray = 255 - gray
        
        # より穏やかなノイズ除去
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        
        # 適応的二値化（カメラ画像に適している）
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 線をわずかに太らせて細線対策
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # リサイズ（アスペクト比を保持、高さは64固定）
        h, w = binary.shape
        aspect_ratio = w / h if h > 0 else 1.0
        new_width = int(target_height * aspect_ratio)
        if new_width > max_width:
            new_width = max_width
        resized = cv2.resize(binary, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # パディング（右側を埋める）
        if new_width < max_width:
            padded = np.zeros((target_height, max_width), dtype=np.uint8)
            padded[:, :new_width] = resized
            resized = padded
        
        # 正規化 (0-1の範囲に)
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
