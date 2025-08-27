import time
import uuid
from io import BytesIO
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image
import torch
import os

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
        
        # デコード・前処理設定（環境変数で切替）
        self.use_beam = os.getenv("CRNN_BEAM_SEARCH", "true").lower() == "true"
        self.beam_width = int(os.getenv("CRNN_BEAM_WIDTH", "5"))
        self.width_scale = float(os.getenv("CRNN_WIDTH_SCALE", "1.0"))  # 横方向の微小拡大
        self.right_margin = int(os.getenv("CRNN_RIGHT_MARGIN", "12"))     # 右側の余白（px）
        self.max_width = int(os.getenv("CRNN_MAX_WIDTH", "256"))
    
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
            env_path = os.getenv("CRNN_MODEL_PATH")
            model_path = Path(env_path) if env_path else (project_root / 'models' / 'crnn_model_best.pth')
            
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
                print(f"CRNN model loaded successfully from {model_path}")
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
            preprocessed = self._preprocess_for_crnn(image, target_height=64, max_width=self.max_width)
            
            # テンソルに変換
            img_tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # 推論
            with torch.no_grad():
                output = self.crnn_model(img_tensor)  # (seq_len, batch, num_classes) in log-softmax
                if self.use_beam:
                    pred_text, avg_confidence = self._ctc_beam_search_decode(output, self.beam_width)
                else:
                    # Greedy fallback
                    _, preds = output.max(2)
                    pred_text = self.crnn_converter.decode(preds)[0]
                    probs = torch.exp(output)
                    max_probs, _ = probs.max(2)
                    confidence_scores = []
                    for i, idx in enumerate(preds[0]):
                        if idx != 0:
                            confidence_scores.append(max_probs[i, 0].item())
                    avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
                
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
        
        # リサイズ（アスペクト比保持、高さ64固定）。横方向の微小拡大を反映
        h, w = binary.shape
        aspect_ratio = (w / h) if h > 0 else 1.0
        new_width = int(target_height * aspect_ratio * max(self.width_scale, 1.0))
        # 右マージンを確保するため、必要なら幅を少し抑える
        max_usable = max_width - max(self.right_margin, 0)
        if max_usable < 1:
            max_usable = max_width
        if new_width > max_usable:
            new_width = max_usable
        if new_width < 1:
            new_width = 1
        resized = cv2.resize(binary, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # パディング（右側に余白を追加）。右マージンを確保して末尾切れを抑制
        padded = np.zeros((target_height, max_width), dtype=np.uint8)
        right_limit = min(new_width + max(self.right_margin, 0), max_width)
        padded[:, :new_width] = resized
        # 右マージン部分はゼロ埋め（背景）
        resized = padded
        
        # 正規化 (0-1の範囲に)
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized

    @staticmethod
    def _log_sum_exp(a: float, b: float) -> float:
        if a == -float('inf'):
            return b
        if b == -float('inf'):
            return a
        m = a if a > b else b
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    def _ctc_beam_search_decode(self, log_probs: torch.Tensor, beam_width: int = 5) -> (str, float):
        """簡易CTCプレフィックスビームサーチ
        Args:
            log_probs: (seq_len, batch=1, num_classes) ログ確率
        Returns:
            best_text, approx_confidence
        """
        blank = 0
        seq_len, batch, num_classes = log_probs.size()
        assert batch == 1
        lp = log_probs[:, 0, :]  # (T, C)

        # beams: prefix -> (p_blank, p_non_blank) in log-domain
        beams = {"": (-0.0, -float('inf'))}  # log(1)=0 as -0.0

        for t in range(seq_len):
            next_beams = {}
            for prefix, (pb, pnb) in beams.items():
                # extend with blank
                p_blank_t = lp[t, blank].item()
                nb_pb = self._log_sum_exp(pb + p_blank_t, pnb + p_blank_t)
                old = next_beams.get(prefix, (-float('inf'), -float('inf')))
                next_beams[prefix] = (self._log_sum_exp(old[0], nb_pb), old[1])

                # extend with characters
                for c in range(1, num_classes):
                    p_t_c = lp[t, c].item()
                    char = self.crnn_converter.idx_to_char.get(c)
                    if not char:
                        continue
                    if len(prefix) > 0 and prefix[-1] == char:
                        # same char: only from blank
                        new_pb, new_pnb = next_beams.get(prefix, (-float('inf'), -float('inf')))
                        new_pnb = self._log_sum_exp(new_pnb, pb + p_t_c)
                        next_beams[prefix] = (new_pb, new_pnb)
                    else:
                        new_prefix = prefix + char
                        old2 = next_beams.get(new_prefix, (-float('inf'), -float('inf')))
                        new_pnb2 = self._log_sum_exp(old2[1], self._log_sum_exp(pb + p_t_c, pnb + p_t_c))
                        next_beams[new_prefix] = (old2[0], new_pnb2)

            # prune
            beams = dict(sorted(next_beams.items(), key=lambda kv: self._log_sum_exp(kv[1][0], kv[1][1]), reverse=True)[:beam_width])

        # find best
        best_prefix = ""
        best_score = -float('inf')
        length_penalty = float(os.getenv("CRNN_BEAM_LENGTH_PENALTY", "1.0"))
        for prefix, (pb, pnb) in beams.items():
            score = self._log_sum_exp(pb, pnb)
            norm = (len(prefix) if len(prefix) > 0 else 1) ** length_penalty
            norm_score = score / norm
            if norm_score > best_score:
                best_score = norm_score
                best_prefix = prefix

        # approximate confidence: use exponentiated normalized score
        approx_conf = float(np.exp(best_score))
        return best_prefix, approx_conf
