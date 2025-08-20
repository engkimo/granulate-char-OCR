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
        # 円検出を使用
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 画像をぼかして円検出の精度を向上
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # HoughCircles で円を検出
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,  # 円の中心間の最小距離（重複を減らすため少し小さく）
            param1=40,   # Cannyエッジ検出の上限閾値（より感度を上げる）
            param2=25,   # 円の中心を検出するための閾値（より多くの円を検出）
            minRadius=35,  # 最小半径（少し小さく）
            maxRadius=75   # 最大半径（少し大きく）
        )
        
        all_bubbles = []
        all_bubble_info = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print(f"検出された円の数: {len(circles[0])}")
            
            # 重複除去: 近い円を除外
            filtered_circles = []
            for i, (x, y, r) in enumerate(circles[0]):
                is_duplicate = False
                for fx, fy, fr in filtered_circles:
                    distance = np.sqrt((x - fx)**2 + (y - fy)**2)
                    if distance < min(r, fr) * 0.8:  # 半径の80%未満の距離なら重複とみなす
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_circles.append((x, y, r))
            
            print(f"重複除去後の円の数: {len(filtered_circles)}")
            
            for i, (x, y, r) in enumerate(filtered_circles):
                # バブル領域を切り出し
                x1 = max(0, x - r - 10)
                y1 = max(0, y - r - 10)
                x2 = min(self.image.shape[1], x + r + 10)
                y2 = min(self.image.shape[0], y + r + 10)
                
                bubble = self.image[y1:y2, x1:x2]
                
                # バブルが紫色かチェック
                hsv = cv2.cvtColor(bubble, cv2.COLOR_BGR2HSV)
                lower_purple = np.array([120, 20, 20])
                upper_purple = np.array([180, 255, 255])
                mask = cv2.inRange(hsv, lower_purple, upper_purple)
                
                purple_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                
                if purple_ratio > 0.3:  # 30%以上が紫色
                    all_bubbles.append(bubble)
                    all_bubble_info.append((x1, y1, x2-x1, y2-y1, np.pi * r * r))
        
        # y座標、次にx座標でソート
        sorted_indices = sorted(range(len(all_bubble_info)), 
                               key=lambda i: (all_bubble_info[i][1], all_bubble_info[i][0]))
        
        sorted_bubbles = [all_bubbles[i] for i in sorted_indices]
        sorted_info = [all_bubble_info[i] for i in sorted_indices]
        
        # グリッドベースの整理（3行×9列を想定）
        if len(sorted_bubbles) > 27:
            # y座標で3つのグループに分割
            y_coords = [info[1] for info in sorted_info]
            y_sorted = sorted(enumerate(y_coords), key=lambda x: x[1])
            
            # 3つの行に分割
            rows_per_group = len(y_sorted) // 3
            row1_indices = [idx for idx, _ in y_sorted[:rows_per_group]]
            row2_indices = [idx for idx, _ in y_sorted[rows_per_group:2*rows_per_group]]
            row3_indices = [idx for idx, _ in y_sorted[2*rows_per_group:]]
            
            # 各行でx座標でソートし、最初の9個を取る
            final_indices = []
            for row_indices in [row1_indices, row2_indices, row3_indices]:
                row_sorted = sorted(row_indices, key=lambda i: sorted_info[i][0])
                final_indices.extend(row_sorted[:9])
            
            # 最終的な27個のバブルを選択
            if len(final_indices) >= 27:
                final_bubbles = [sorted_bubbles[i] for i in final_indices[:27]]
                final_info = [sorted_info[i] for i in final_indices[:27]]
                
                print(f"グリッドベースで選択されたバブル数: {len(final_bubbles)}")
                return final_bubbles, final_info
        
        return sorted_bubbles, sorted_info
    
    def _detect_rows(self) -> List[Tuple[int, int]]:
        """画像から行を検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # 紫色の範囲を定義（より広めに）
        lower_purple = np.array([120, 20, 20])
        upper_purple = np.array([180, 255, 255])
        
        # 紫色のマスクを作成
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 水平方向の投影
        horizontal_projection = np.sum(mask, axis=1)
        
        # デバッグ出力
        print(f"マスク内のピクセル数: {np.sum(mask > 0)}")
        print(f"最大投影値: {np.max(horizontal_projection)}")
        
        # 行の境界を検出
        rows = []
        in_row = False
        row_start = 0
        
        threshold = np.max(horizontal_projection) * 0.05  # より低い閾値
        
        for y, value in enumerate(horizontal_projection):
            if not in_row and value > threshold:
                in_row = True
                row_start = y
            elif in_row and value <= threshold:
                in_row = False
                if y - row_start > 30:  # より低い最小行高さ
                    rows.append((row_start - 10, y - row_start + 20))  # 余白を追加
        
        # 最後の行も追加
        if in_row and len(horizontal_projection) - row_start > 30:
            rows.append((row_start - 10, len(horizontal_projection) - row_start + 10))
        
        print(f"検出された行数: {len(rows)}")
        for i, (y, h) in enumerate(rows):
            print(f"行{i}: y={y}, height={h}")
        
        return rows
    
    def _detect_individual_bubbles(self, row_img: np.ndarray, row_y: int) -> Tuple[List[np.ndarray], List[Tuple]]:
        """行内の個別バブルを検出"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
        
        # 紫色の範囲を定義（より広めに）
        lower_purple = np.array([120, 20, 20])
        upper_purple = np.array([180, 255, 255])
        
        # 紫色のマスクを作成
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # 円形のカーネルでモルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # デバッグ出力
        print(f"行内のマスクピクセル数: {np.sum(mask > 0)}")
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"検出された輪郭数: {len(contours)}")
        
        bubbles = []
        bubble_info = []
        
        # 円形度でフィルタリング
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # デバッグ出力
            if i < 5:  # 最初の5つの輪郭情報を出力
                print(f"輪郭{i}: 面積={area}")
            
            if area < 500:  # より低い閾値
                continue
            
            # 円形度を計算
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # デバッグ出力
            if area > 500:
                print(f"候補: 面積={area}, 円形度={circularity:.2f}, アスペクト比={aspect_ratio:.2f}")
            
            # より緩い条件
            if circularity > 0.4 and 0.7 < aspect_ratio < 1.3:
                bubble = row_img[y:y+h, x:x+w]
                bubbles.append(bubble)
                bubble_info.append((x, row_y + y, w, h, area))
        
        # x座標でソート
        sorted_indices = sorted(range(len(bubble_info)), key=lambda i: bubble_info[i][0])
        
        sorted_bubbles = [bubbles[i] for i in sorted_indices]
        sorted_info = [bubble_info[i] for i in sorted_indices]
        
        print(f"抽出されたバブル数: {len(sorted_bubbles)}")
        
        return sorted_bubbles, sorted_info
    
    def extract_alphabet_label(self, bubble_image: np.ndarray) -> str:
        """バブルの右上から黄色のアルファベットを抽出してOCR"""
        # HSV色空間に変換
        hsv = cv2.cvtColor(bubble_image, cv2.COLOR_BGR2HSV)
        
        # 黄色の範囲を定義（より広い範囲）
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([45, 255, 255])
        
        # 黄色のマスクを作成
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 右上の領域に限定（バブルの右上半分）
        h, w = bubble_image.shape[:2]
        roi_mask = np.zeros_like(yellow_mask)
        roi_mask[0:h//2, w//3:] = 255
        
        # マスクを結合
        yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)
        
        # ノイズ除去
        kernel = np.ones((2, 2), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        
        # 輪郭検出
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を取得
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 面積が小さすぎる場合はスキップ
        if w * h < 50:
            return None
        
        # アルファベット領域を切り出し
        padding = 8
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(bubble_image.shape[1], x + w + padding)
        y_end = min(bubble_image.shape[0], y + h + padding)
        
        alphabet_region = bubble_image[y_start:y_end, x_start:x_end]
        
        # 画像を拡大してOCR精度を向上
        scale_factor = 4
        alphabet_region = cv2.resize(alphabet_region, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # グレースケール変換
        gray = cv2.cvtColor(alphabet_region, cv2.COLOR_BGR2GRAY)
        
        # 複数の二値化方法を試す
        results = []
        
        # 方法1: 通常のOTSU
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法2: 適応的閾値処理
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # 方法3: 黄色のマスクから直接
        yellow_region = yellow_mask[y_start:y_end, x_start:x_end]
        yellow_region = cv2.resize(yellow_region, None, fx=scale_factor, fy=scale_factor, 
                                  interpolation=cv2.INTER_CUBIC)
        
        # OCRでアルファベットを認識
        for i, binary in enumerate([binary1, binary2, yellow_region]):
            try:
                # 文字だけを認識するようPSM 10を使用
                custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                text = pytesseract.image_to_string(binary, config=custom_config).strip()
                
                # 1文字のアルファベットのみを返す
                if len(text) == 1 and text.isalpha():
                    results.append(text.upper())
            except Exception as e:
                continue
        
        # 最も頻出する結果を返す
        if results:
            from collections import Counter
            most_common = Counter(results).most_common(1)[0][0]
            return most_common
        
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
            
            # 手動マッピングを試行
            manual_mappings = self._try_manual_mapping(failed_ocr, bubbles, output_path, char_mapping)
            extracted_count += len(manual_mappings)
            char_mapping.update(manual_mappings)
        
        # マッピング情報を保存
        mapping_file = output_path / "character_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(char_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"\n抽出完了: {extracted_count}/{26} 文字")
        print(f"マッピングファイル: {mapping_file}")
        
        return char_mapping
    
    def _try_manual_mapping(self, failed_indices: List[int], bubbles: List[np.ndarray], 
                          output_path: Path, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        """OCRに失敗したバブルを手動でマッピング"""
        manual_mapping = {}
        
        # 既に抽出された文字を取得
        extracted_chars = set(existing_mapping.keys())
        all_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        missing_chars = sorted(all_chars - extracted_chars)
        
        print(f"\n不足している文字: {missing_chars}")
        
        # グリッドベースで27個選択後の、全26文字のマッピング
        # 3行×9列のレイアウト
        complete_char_mapping = [
            # 1行目: A B C D E F G H I
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            # 2行目: J K L M N O P Q R  
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
            # 3行目: S T U V W X Y Z (最後の1つは数字なので除外)
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        # 失敗したインデックスとそれに対応する文字をマッピング
        for idx in failed_indices:
            if idx < len(complete_char_mapping) and idx < len(bubbles):
                char = complete_char_mapping[idx]
                if char in missing_chars:
                    bubble = bubbles[idx]
                    granulate_char = self.extract_granulate_character(bubble)
                    
                    if granulate_char is not None:
                        # 保存ディレクトリを作成
                        char_dir = output_path / char
                        char_dir.mkdir(exist_ok=True)
                        
                        # 画像を保存
                        filename = f"{char}_reference.png"
                        cv2.imwrite(str(char_dir / filename), granulate_char)
                        
                        # バブル画像も保存
                        cv2.imwrite(str(char_dir / f"{char}_bubble.png"), bubble)
                        
                        manual_mapping[char] = str(char_dir / filename)
                        print(f"手動マッピング: インデックス {idx} → {char}")
        
        # まだ不足している文字がある場合、追加でマッピング
        remaining_missing = sorted(set(missing_chars) - set(manual_mapping.keys()))
        if remaining_missing:
            print(f"\nまだ不足している文字: {remaining_missing}")
            # 各文字のインデックスを見つけて追加
            for char in remaining_missing:
                if char in complete_char_mapping:
                    idx = complete_char_mapping.index(char)
                    if idx < len(bubbles):
                        bubble = bubbles[idx]
                        granulate_char = self.extract_granulate_character(bubble)
                        
                        if granulate_char is not None:
                            # 保存ディレクトリを作成
                            char_dir = output_path / char
                            char_dir.mkdir(exist_ok=True)
                            
                            # 画像を保存  
                            filename = f"{char}_reference.png"
                            cv2.imwrite(str(char_dir / filename), granulate_char)
                            
                            # バブル画像も保存
                            cv2.imwrite(str(char_dir / f"{char}_bubble.png"), bubble)
                            
                            manual_mapping[char] = str(char_dir / filename)
                            print(f"追加マッピング: インデックス {idx} → {char}")
        
        return manual_mapping
    
    def visualize_extraction(self):
        """抽出プロセスを視覚化"""
        # 円検出を使用
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=40,
            param2=25,
            minRadius=35,
            maxRadius=75
        )
        
        # 検出結果を描画
        result_image = self.image.copy()
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for i, (x, y, r) in enumerate(circles[0]):
                # 円を描画
                cv2.circle(result_image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
                
                # バブル領域を切り出してアルファベットを認識
                x1 = max(0, x - r - 10)
                y1 = max(0, y - r - 10)
                x2 = min(self.image.shape[1], x + r + 10)
                y2 = min(self.image.shape[0], y + r + 10)
                
                bubble = self.image[y1:y2, x1:x2]
                alphabet = self.extract_alphabet_label(bubble)
                
                if alphabet:
                    cv2.putText(result_image, alphabet, (x-10, y-r-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(result_image, f"?{i}", (x-10, y-r-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 結果を保存
        output_path = Path("training_data/extraction_debug.png")
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        
        print(f"検出結果を保存: {output_path}")
        print(f"検出された円の数: {len(circles[0]) if circles is not None else 0}")


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