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
import os

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
        # CRNN優先のため、CNNはロードしない（必要時のみフォールバックする実装にする場合に備える）
        super().__init__(load_cnn=False)
        self.crnn_model = None
        self.crnn_converter = None
        # デバイス選択（環境変数 TORCH_DEVICE / CRNN_DEVICE を優先）
        self.device = self._select_device()
        self._load_crnn_model()
        print("Initialized OCRServiceWithCRNN (CNN disabled, CRNN primary)")
        # デコード/前処理オプション（環境変数で切替）
        self.use_beam = os.getenv("CRNN_BEAM_SEARCH", "true").lower() == "true"
        self.beam_width = int(os.getenv("CRNN_BEAM_WIDTH", "5"))
        self.width_scale = float(os.getenv("CRNN_WIDTH_SCALE", "1.0"))
        self.right_margin = int(os.getenv("CRNN_RIGHT_MARGIN", "12"))
        self.max_width = int(os.getenv("CRNN_MAX_WIDTH", "256"))
        # 語彙制約
        self.lexicon, self.lexicon_prefixes = self._load_lexicon()
        self.lexicon_strict = os.getenv("CRNN_LEXICON_STRICT", "true").lower() == "true"
        self.lexicon_bonus = float(os.getenv("CRNN_LEXICON_BONUS", "1.0"))
        # 簡易言語モデル（unigram/bigram）
        self.lm_unigram, self.lm_bigram, self.lm_weight = self._load_char_lm()
        # 混同対策（Eに寄りがち対策）
        self.confusion_tweaks = os.getenv("CRNN_CONFUSION_TWEAKS", "true").lower() == "true"
        self.e_penalty = float(os.getenv("CRNN_E_PENALTY", "0.95"))  # 乗算係数
        self.rdn_boost = float(os.getenv("CRNN_RDN_BOOST", "1.05"))  # 乗算係数
        # 語彙オートコレクト
        self.lexicon_autocorrect = os.getenv("CRNN_LEXICON_AUTOCORRECT", "false").lower() == "true"
        self.lexicon_autocorrect_min_ratio = float(os.getenv("CRNN_LEXICON_AUTOCORRECT_MIN_RATIO", "0.85"))
    
    def _load_crnn_model(self):
        """CRNNモデルをロード"""
        try:
            env_path = os.getenv("CRNN_MODEL_PATH")
            model_path = Path(env_path) if env_path else (project_root / 'models' / 'crnn_model_best.pth')
            
            if model_path.exists():
                # 環境変数から文字集合を取得しモデル/コンバータを整合
                charset = os.getenv("CRNN_CHARSET", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                self.crnn_model = create_crnn_model(character_set=charset)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
                self.crnn_model.to(self.device)
                self.crnn_model.eval()
                self.crnn_converter = CTCLabelConverter(character_set=charset)
                print(f"CRNN model loaded successfully from {model_path}")
            else:
                print(f"CRNN model not found at {model_path}")
        except Exception as e:
            print(f"Error loading CRNN model: {e}")
            self.crnn_model = None

    def _select_device(self) -> torch.device:
        """TORCH_DEVICE/CRNN_DEVICEがあればそれを使用。なければMPS→CUDA→CPUの順。"""
        override = os.getenv("CRNN_DEVICE") or os.getenv("TORCH_DEVICE")
        if override:
            try:
                dev = torch.device(override)
                print(f"Using device (override): {dev}")
                return dev
            except Exception:
                print(f"Invalid device override: {override}. Falling back to auto-detect.")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Using device: mps")
            return torch.device('mps')
        if torch.cuda.is_available():
            print("Using device: cuda")
            return torch.device('cuda')
        print("Using device: cpu")
        return torch.device('cpu')
    
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
            preprocessed = self._preprocess_for_crnn(image, target_height=64, max_width=self.max_width)
            
            # テンソルに変換
            img_tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).float()
            img_tensor = img_tensor.to(self.device)
            
            # 推論
            with torch.no_grad():
                output = self.crnn_model(img_tensor)
                if self.use_beam:
                    pred_text, _ = self._ctc_beam_search_decode(output, self.beam_width)
                else:
                    _, preds = output.max(2)
                    pred_text = self.crnn_converter.decode(preds)[0]
                
                # 空文字列の場合はNoneを返す
                if not pred_text:
                    return None
                # 語彙オートコレクト（任意）
                if self.lexicon_autocorrect and self.lexicon:
                    pred_text = self._autocorrect_with_lexicon(pred_text)

                return pred_text
                
        except Exception as e:
            print(f"CRNN processing error: {e}")
            return None
    
    def _preprocess_for_crnn(self, image: np.ndarray, target_height: int = 64, max_width: int = 256) -> np.ndarray:
        """CRNN用の前処理（入力高さは常に64に固定）"""
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
        
        # 二値化（Otsu）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # カメラ画像対応：細い線を軽く太らせる（過学習を避けるため軽め）
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # リサイズ（高さ64固定・横方向スケール適用）
        h, w = binary.shape
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio * max(self.width_scale, 1.0))
        # 右マージンを確保するために必要なら幅を抑える
        max_usable = max_width - max(self.right_margin, 0)
        if max_usable < 1:
            max_usable = max_width
        if new_width > max_usable:
            new_width = max_usable
        if new_width < 1:
            new_width = 1
        binary = cv2.resize(binary, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # パディング（右側に余白を追加）
        padded = np.zeros((target_height, max_width), dtype=np.uint8)
        padded[:, :new_width] = binary
        
        # 正規化
        padded = padded.astype(np.float32) / 255.0
        
        return padded

    @staticmethod
    def _log_sum_exp(a: float, b: float) -> float:
        if a == -float('inf'):
            return b
        if b == -float('inf'):
            return a
        m = a if a > b else b
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    def _ctc_beam_search_decode(self, log_probs: torch.Tensor, beam_width: int = 5):
        """簡易CTCプレフィックスビームサーチ
        Args:
            log_probs: (seq_len, batch=1, num_classes) ログ確率
        Returns:
            best_text, approx_confidence
        """
        blank = 0
        seq_len, batch, num_classes = log_probs.size()
        assert batch == 1
        lp = log_probs[:, 0, :]
        beams = {"": (-0.0, -float('inf'))}
        for t in range(seq_len):
            next_beams = {}
            for prefix, (pb, pnb) in beams.items():
                p_blank_t = lp[t, blank].item()
                nb_pb = self._log_sum_exp(pb + p_blank_t, pnb + p_blank_t)
                old = next_beams.get(prefix, (-float('inf'), -float('inf')))
                next_beams[prefix] = (self._log_sum_exp(old[0], nb_pb), old[1])
                for c in range(1, num_classes):
                    p_t_c = lp[t, c].item()
                    char = self.crnn_converter.idx_to_char.get(c)
                    if not char:
                        continue
                    # 混同対策とLM重み付けを事前に加点（ログ領域）
                    bonus = 0.0
                    # 文字言語モデル（前文字に依存）
                    if self.lm_weight > 0.0:
                        prev_char = prefix[-1] if len(prefix) > 0 else None
                        bonus += self.lm_weight * self._lm_log_prob(prev_char, char)
                    # E←→{R,D,N}の傾向補正
                    if self.confusion_tweaks:
                        if char == 'E' and self.e_penalty > 0 and self.e_penalty != 1.0:
                            bonus += float(np.log(max(self.e_penalty, 1e-6)))
                        elif char in {'R', 'D', 'N'} and self.rdn_boost > 0 and self.rdn_boost != 1.0:
                            bonus += float(np.log(self.rdn_boost))
                    p_t_c = p_t_c + bonus
                    if len(prefix) > 0 and prefix[-1] == char:
                        if not self.lexicon_strict or self._is_valid_prefix(prefix):
                            new_pb, new_pnb = next_beams.get(prefix, (-float('inf'), -float('inf')))
                            new_pnb = self._log_sum_exp(new_pnb, pb + p_t_c)
                            next_beams[prefix] = (new_pb, new_pnb)
                    else:
                        new_prefix = prefix + char
                        if not self.lexicon_strict or self._is_valid_prefix(new_prefix):
                            old2 = next_beams.get(new_prefix, (-float('inf'), -float('inf')))
                            new_pnb2 = self._log_sum_exp(old2[1], self._log_sum_exp(pb + p_t_c, pnb + p_t_c))
                            next_beams[new_prefix] = (old2[0], new_pnb2)
            beams = dict(sorted(next_beams.items(), key=lambda kv: self._log_sum_exp(kv[1][0], kv[1][1]), reverse=True)[:beam_width])
        best_prefix = ""
        best_score = -float('inf')
        length_penalty = float(os.getenv("CRNN_BEAM_LENGTH_PENALTY", "1.0"))
        for prefix, (pb, pnb) in beams.items():
            score = self._log_sum_exp(pb, pnb)
            norm = (len(prefix) if len(prefix) > 0 else 1) ** length_penalty
            norm_score = score / norm
            if self.lexicon and prefix in self.lexicon and self.lexicon_bonus > 1.0:
                norm_score += float(np.log(self.lexicon_bonus))
            if norm_score > best_score:
                best_score = norm_score
                best_prefix = prefix
        approx_conf = float(np.exp(best_score))
        return best_prefix, approx_conf

    def _load_lexicon(self):
        path = os.getenv("CRNN_LEXICON_PATH")
        inline = os.getenv("CRNN_LEXICON")
        words = set()
        try:
            if path and Path(path).exists():
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        w = line.strip().upper()
                        if w and w.isalpha():
                            words.add(w)
            elif inline:
                for w in inline.split(','):
                    w = w.strip().upper()
                    if w and w.isalpha():
                        words.add(w)
        except Exception as e:
            print(f"Lexicon load error: {e}")
        prefixes = set()
        for w in words:
            for i in range(1, len(w)+1):
                prefixes.add(w[:i])
        if words:
            print(f"Lexicon loaded: {len(words)} words")
        return words, prefixes

    def _is_valid_prefix(self, s: str) -> bool:
        if not self.lexicon_prefixes:
            return True
        return s in self.lexicon_prefixes

    def _load_char_lm(self):
        """単純な文字言語モデル（unigram/bigram）をJSONから読み込み
        JSON例:
        {"unigram": {"E":0.127, "T":0.091, ...}, "bigram": {"TH":0.02, "HE":0.018, ...}, "weight": 0.2}
        戻り値は (unigram_log, bigram_log, weight)
        """
        path = os.getenv("CRNN_LM_PATH")
        weight = float(os.getenv("CRNN_LM_WEIGHT", "0.0"))
        uni_log, bi_log = None, None
        if path and Path(path).exists():
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                uni = data.get('unigram') or {}
                bi = data.get('bigram') or {}
                # 正規化してログ確率に
                def to_logprob(d):
                    # 数値を合計1にし、logを取る。未定義は均一(極小)扱い
                    if not d:
                        return None
                    total = float(sum(max(v, 0.0) for v in d.values())) or 1.0
                    return {k.upper(): float(np.log(max(v, 1e-12) / total)) for k, v in d.items()}
                uni_log = to_logprob(uni)
                # bigramキーは2文字
                if bi:
                    total_bi = float(sum(max(v, 0.0) for v in bi.values())) or 1.0
                    bi_log = {k.upper(): float(np.log(max(v, 1e-12) / total_bi)) for k, v in bi.items()}
                if 'weight' in data:
                    weight = float(data['weight'])
                print(f"Char LM loaded (weight={weight}) from {path}")
            except Exception as e:
                print(f"Char LM load error: {e}")
        return uni_log, bi_log, weight

    def _lm_log_prob(self, prev_char: Optional[str], curr_char: str) -> float:
        """LMの対数確率（正規化済み）を返す。未定義は0（等確率）"""
        if prev_char and self.lm_bigram:
            key = f"{prev_char}{curr_char}"
            if key in self.lm_bigram:
                return self.lm_bigram[key]
        if self.lm_unigram and curr_char in self.lm_unigram:
            return self.lm_unigram[curr_char]
        return 0.0

    def _autocorrect_with_lexicon(self, text: str) -> str:
        """difflibの類似度で語彙内の最も近い単語に補正（閾値以上の場合のみ）"""
        try:
            import difflib
            best = text
            best_ratio = 0.0
            # 文字数が極端に違う語はスキップして計算量を削減
            for w in self.lexicon:
                if abs(len(w) - len(text)) > 3:
                    continue
                ratio = difflib.SequenceMatcher(a=text, b=w).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = w
            if best_ratio >= self.lexicon_autocorrect_min_ratio:
                return best
            return text
        except Exception:
            return text
