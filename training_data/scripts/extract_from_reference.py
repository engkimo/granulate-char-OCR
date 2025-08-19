import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pytesseract


class GranulateCharacterExtractor:
    """早見表からグラニュート文字を抽出するクラス"""
    
    def __init__(self, reference_image_path: str):
        self.image_path = Path(reference_image_path)
        self.image = cv2.imread(str(self.image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # 既知の文字マッピング（画像の順番に基づく）
        self.character_order = [
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        ]
        
    def extract_purple_bubbles(self) -> Tuple[List[np.ndarray], List[Tuple]]:
        """紫色のバブルを検出して個別に抽出"""
        # まず行を検出
        rows = self._detect_rows()
        
        all_bubbles = []
        all_bubble_info = []
        
        # 各行内で個別のバブルを検出
        for row_idx, (row_y, row_h) in enumerate(rows):
            # 行の領域を切り出し
            row_img = self.image[row_y:row_y+row_h, :]
            
            # 個別のバブルを検出
            bubbles, bubble_info = self._detect_individual_bubbles(row_img, row_y)
            
            all_bubbles.extend(bubbles)
            all_bubble_info.extend(bubble_info)
        
        return all_bubbles, all_bubble_info
    
    def _detect_rows(self) -> List[Tuple[int, int]]:
        """画像から行を検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # 紫色の範囲を定義
        lower_purple = np.array([130, 30, 30])
        upper_purple = np.array([170, 255, 255])
        
        # 紫色のマスクを作成
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 水平方向の投影
        horizontal_projection = np.sum(mask, axis=1)
        
        # 行の境界を検出
        rows = []
        in_row = False
        row_start = 0
        
        threshold = np.max(horizontal_projection) * 0.1
        
        for y, value in enumerate(horizontal_projection):
            if not in_row and value > threshold:
                in_row = True
                row_start = y
            elif in_row and value <= threshold:
                in_row = False
                if y - row_start > 50:  # 最小行高さ
                    rows.append((row_start - 10, y - row_start + 20))  # 余白を追加
        
        return rows
    
    def _detect_individual_bubbles(self, row_img: np.ndarray, row_y: int) -> Tuple[List[np.ndarray], List[Tuple]]:
        """行内の個別バブルを検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        
        # 紫色の範囲を定義（より厳密に）
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([170, 255, 255])
        
        # 紫色のマスクを作成
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 円形のカーネルでモルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        bubble_info = []
        
        # 円形度でフィルタリング
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # 小さすぎる輪郭は除外
                continue
            
            # 円形度を計算
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            if circularity > 0.6:  # 円形に近いもののみ
                x, y, w, h = cv2.boundingRect(contour)
                
                # アスペクト比チェック（正方形に近いか）
                aspect_ratio = w / h if h > 0 else 0
                if 0.8 < aspect_ratio < 1.2:
                    bubble = row_img[y:y+h, x:x+w]
                    bubbles.append(bubble)
                    bubble_info.append((x, row_y + y, w, h, area))
        
        # x座標でソート
        sorted_indices = sorted(range(len(bubble_info)), key=lambda i: bubble_info[i][0])
        
        sorted_bubbles = [bubbles[i] for i in sorted_indices]
        sorted_info = [bubble_info[i] for i in sorted_indices]
        
        return sorted_bubbles, sorted_info
    
    def extract_alphabet_label(self, bubble_image: np.ndarray) -> str:
        """バブルの右上から黄色のアルファベットを抽出してOCR"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2HSV)
        
        # 黄色の範囲を定義
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # 黄色のマスクを作成
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 右上の領域に限定（バブルの右上1/3）
        h, w = bubble_image.shape[:2]
        roi_mask = np.zeros_like(yellow_mask)
        roi_mask[0:h//2, w//2:] = 255
        
        # マスクを結合
        yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)
        
        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # アルファベット領域を切り出し
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(bubble_image.shape[1], x + w + padding)
        y_end = min(bubble_image.shape[0], y + h + padding)
        
        alphabet_region = bubble_image[y_start:y_end, x_start:x_end]
        
        # 画像を拡大してOCR精度を向上
        scale_factor = 3
        alphabet_region = cv2.resize(alphabet_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # グレースケール変換して二値化
        gray = cv2.cvtColor(alphabet_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCRでアルファベットを認識
        try:
            custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            
            # 1文字のアルファベットのみを返す
            if len(text) == 1 and text.isalpha():
                return text.upper()
        except Exception as e:
            print(f"OCRエラー: {e}")
        
        return None
    
    def extract_granulate_character(self, bubble_image: np.ndarray) -> np.ndarray:
        """バブルからグラニュート文字部分を抽出"""
        # グレースケール変換
        gray = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2GRAY)
        
        # 白い文字を検出（グラニュート文字は白色）
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 中央付近の領域に限定
        h, w = bubble_image.shape[:2]
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        
        # 文字領域を検出
        contours, _ = cv2.findContours(center_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を文字として扱う
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 元の座標に変換
        x += bubble_image.shape[1] // 4
        y += bubble_image.shape[0] // 4
        
        # 余白を追加してクロップ
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(bubble_image.shape[1], x + w + padding)
        y_end = min(bubble_image.shape[0], y + h + padding)
        
        character = binary[y_start:y_end, x_start:x_end]
        
        # サイズを正規化（64x64）
        if character.size > 0:
            character_resized = cv2.resize(character, (64, 64), interpolation=cv2.INTER_CUBIC)
            return character_resized
        
        return None
    
    def process_all_characters(self, output_dir: str = "training_data/extracted"):
        """全文字を抽出して保存"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # バブルを抽出
        bubbles, bubble_info = self.extract_purple_bubbles()
        
        print(f"検出されたバブル数: {len(bubbles)}")
        
        # 文字マッピング
        char_mapping = {}
        extracted_count = 0
        failed_ocr = []
        
        # 各バブルを処理
        for idx, bubble in enumerate(bubbles):
            # 黄色のアルファベットをOCRで認識
            alphabet = self.extract_alphabet_label(bubble)
            
            if alphabet:
                # グラニュート文字を抽出
                granulate_char = self.extract_granulate_character(bubble)
                
                if granulate_char is not None:
                    # 保存ディレクトリを作成
                    char_dir = output_path / alphabet
                    char_dir.mkdir(exist_ok=True)
                    
                    # 画像を保存
                    filename = f"{alphabet}_reference.png"
                    cv2.imwrite(str(char_dir / filename), granulate_char)
                    
                    # 視覚化用にも保存
                    plt.figure(figsize=(3, 3))
                    plt.imshow(granulate_char, cmap='gray')
                    plt.title(f"Granulate: {alphabet}")
                    plt.axis('off')
                    plt.savefig(str(char_dir / f"{alphabet}_preview.png"), 
                               bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    # バブル画像も保存（デバッグ用）
                    cv2.imwrite(str(char_dir / f"{alphabet}_bubble.png"), bubble)
                    
                    extracted_count += 1
                    char_mapping[alphabet] = str(char_dir / filename)
                    
                    print(f"抽出完了: {alphabet}")
                else:
                    print(f"警告: バブル{idx}からグラニュート文字を抽出できませんでした（アルファベット: {alphabet}）")
            else:
                failed_ocr.append(idx)
                # OCRが失敗した場合、バブル画像を保存してデバッグ
                debug_dir = output_path / "debug"
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / f"failed_ocr_bubble_{idx}.png"), bubble)
        
        if failed_ocr:
            print(f"\nOCRに失敗したバブル: {failed_ocr}")
            print("デバッグ画像が training_data/extracted/debug/ に保存されました")
        
        # マッピング情報を保存
        mapping_file = output_path / "character_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(char_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"\n抽出完了: {extracted_count}/{26} 文字")
        print(f"マッピングファイル: {mapping_file}")
        
        return char_mapping
    
    def visualize_extraction(self):
        """抽出プロセスを視覚化"""
        bubbles, bubble_info = self.extract_purple_bubbles()
        
        # 検出結果を描画
        result_image = self.image.copy()
        
        for i, (x, y, w, h, _) in enumerate(bubble_info):
            # バブルの矩形を描画
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # バブルから黄色のアルファベットを認識
            if i < len(bubbles):
                alphabet = self.extract_alphabet_label(bubbles[i])
                if alphabet:
                    # アルファベットを表示
                    cv2.putText(result_image, alphabet, (x+5, y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    # OCR失敗時はインデックスを表示
                    cv2.putText(result_image, f"?{i}", (x+5, y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 結果を保存
        output_path = Path("training_data/extraction_debug.png")
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        
        print(f"検出結果を保存: {output_path}")
        print(f"総バブル数: {len(bubbles)}")


if __name__ == "__main__":
    # Tesseractが利用可能か確認
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        print(f"エラー: Tesseractがインストールされていません。")
        print(f"Macの場合: brew install tesseract")
        print(f"Ubuntuの場合: sudo apt-get install tesseract-ocr")
        exit(1)
    
    # 早見表から文字を抽出
    extractor = GranulateCharacterExtractor("static/granulte_chars.jpg")
    
    # デバッグ用視覚化
    extractor.visualize_extraction()
    
    # 全文字を抽出
    mapping = extractor.process_all_characters()
    
    print("\n抽出された文字:")
    for char, path in sorted(mapping.items()):
        print(f"{char}: {path}")