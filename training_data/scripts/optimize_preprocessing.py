#!/usr/bin/env python3
"""
画像前処理の最適化
グラニュート文字認識の精度向上のための前処理パイプライン
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm


class ImagePreprocessor:
    """画像前処理クラス"""
    
    def __init__(self):
        self.methods = {
            'baseline': self.baseline_preprocessing,
            'adaptive_threshold': self.adaptive_threshold_preprocessing,
            'morphological': self.morphological_preprocessing,
            'edge_preserving': self.edge_preserving_preprocessing,
            'combined': self.combined_preprocessing,
            'denoise_sharp': self.denoise_and_sharpen,
            'contrast_enhance': self.contrast_enhancement
        }
    
    def baseline_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """ベースライン前処理（単純な二値化）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 単純な二値化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def adaptive_threshold_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """適応的閾値処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ガウシアンブラー（ノイズ除去）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 適応的閾値処理
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return adaptive
    
    def morphological_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """モルフォロジー処理を含む前処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ノイズ除去
        denoised = cv2.medianBlur(gray, 3)
        
        # 適応的閾値処理
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # モルフォロジー処理
        kernel = np.ones((2, 2), np.uint8)
        
        # クロージング（穴を埋める）
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # オープニング（ノイズ除去）
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def edge_preserving_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """エッジ保存フィルタを使用した前処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # エッジ保存フィルタ
        if len(image.shape) == 3:
            filtered = cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.4)
            gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        else:
            # グレースケールの場合は3チャンネルに変換してフィルタ適用
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            filtered = cv2.edgePreservingFilter(rgb, flags=2, sigma_s=50, sigma_r=0.4)
            gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        
        # CLAHE（コントラスト制限適応ヒストグラム均等化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_filtered)
        
        # 二値化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def combined_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """複合的な前処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. ノイズ除去
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. シャープニング
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 4. 適応的二値化
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. モルフォロジー処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def denoise_and_sharpen(self, image: np.ndarray) -> np.ndarray:
        """ノイズ除去とシャープニング"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # アンシャープマスク
        gaussian = cv2.GaussianBlur(denoised, (5, 5), 1.0)
        sharpened = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        
        # 二値化
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """コントラスト強調に特化した前処理"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ヒストグラム均等化
        equalized = cv2.equalizeHist(gray)
        
        # ガンマ補正
        gamma = 1.2
        gamma_corrected = np.power(equalized / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        # 適応的二値化
        binary = cv2.adaptiveThreshold(
            gamma_corrected, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        
        return binary
    
    def process_image(self, image: np.ndarray, method: str) -> np.ndarray:
        """指定された方法で画像を前処理"""
        if method in self.methods:
            return self.methods[method](image)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")


def evaluate_preprocessing_methods(data_dir: Path, test_model_func=None):
    """各前処理手法を評価"""
    preprocessor = ImagePreprocessor()
    results = {}
    
    # テスト画像を収集
    test_images = []
    test_labels = []
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        char_dir = data_dir / char
        if char_dir.exists():
            # 各文字から5枚ずつテスト
            images = list(char_dir.glob("*.png"))[:5]
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is not None:
                    test_images.append(img)
                    test_labels.append(char)
    
    print(f"Testing {len(test_images)} images with different preprocessing methods...\n")
    
    # 各前処理手法を評価
    for method_name in preprocessor.methods.keys():
        print(f"Testing {method_name}...")
        
        method_results = {
            'sample_images': [],
            'processing_times': [],
            'quality_scores': []
        }
        
        for img in tqdm(test_images[:10]):  # サンプル画像を保存
            # 前処理を適用
            processed = preprocessor.process_image(img, method_name)
            method_results['sample_images'].append(processed)
            
            # 画像品質スコアを計算（エッジの鮮明さ）
            edges = cv2.Canny(processed, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            method_results['quality_scores'].append(edge_density)
        
        # 平均品質スコア
        avg_quality = np.mean(method_results['quality_scores'])
        
        results[method_name] = {
            'avg_quality_score': avg_quality,
            'sample_images': method_results['sample_images'][:5]
        }
        
        print(f"  Average quality score: {avg_quality:.4f}")
    
    return results


def visualize_preprocessing_comparison(results: Dict, output_path: Path):
    """前処理手法の比較を可視化"""
    methods = list(results.keys())
    n_methods = len(methods)
    n_samples = 3  # 表示するサンプル数
    
    fig, axes = plt.subplots(n_methods, n_samples + 1, figsize=(15, 3 * n_methods))
    
    # 品質スコアをプロット
    quality_scores = [results[m]['avg_quality_score'] for m in methods]
    
    for i, method in enumerate(methods):
        # 品質スコアバー
        axes[i, 0].barh([0], [results[method]['avg_quality_score']], color='blue')
        axes[i, 0].set_xlim(0, max(quality_scores) * 1.2)
        axes[i, 0].set_title(f"{method}\nScore: {results[method]['avg_quality_score']:.3f}")
        axes[i, 0].set_yticks([])
        
        # サンプル画像
        for j in range(n_samples):
            if j < len(results[method]['sample_images']):
                axes[i, j + 1].imshow(results[method]['sample_images'][j], cmap='gray')
                axes[i, j + 1].axis('off')
                if i == 0:
                    axes[i, j + 1].set_title(f"Sample {j + 1}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nComparison visualization saved to: {output_path}")


def create_optimized_preprocessor():
    """最適化された前処理パイプラインを作成"""
    
    class OptimizedPreprocessor:
        """グラニュート文字認識に最適化された前処理"""
        
        def __init__(self):
            self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        def process(self, image: np.ndarray) -> np.ndarray:
            """最適化された前処理パイプライン"""
            # グレースケール変換
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. ノイズ除去（バイラテラルフィルタ）
            denoised = cv2.bilateralFilter(gray, 5, 50, 50)
            
            # 2. コントラスト強調（CLAHE）
            enhanced = self.clahe.apply(denoised)
            
            # 3. シャープニング（控えめ）
            kernel = np.array([[0, -0.5, 0],
                             [-0.5, 3, -0.5],
                             [0, -0.5, 0]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. 適応的二値化
            binary = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3
            )
            
            # 5. 細かいノイズ除去
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 6. エッジを滑らかに
            final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            return final
    
    return OptimizedPreprocessor()


def main():
    # データディレクトリ
    data_dir = Path("training_data/augmented")
    output_dir = Path("preprocessing_results")
    output_dir.mkdir(exist_ok=True)
    
    # 前処理手法を評価
    print("=== Evaluating Preprocessing Methods ===\n")
    results = evaluate_preprocessing_methods(data_dir)
    
    # 結果を可視化
    visualize_preprocessing_comparison(results, output_dir / "preprocessing_comparison.png")
    
    # 最適な手法を選択
    best_method = max(results.items(), key=lambda x: x[1]['avg_quality_score'])[0]
    print(f"\nBest preprocessing method: {best_method}")
    print(f"Quality score: {results[best_method]['avg_quality_score']:.4f}")
    
    # 結果を保存
    summary = {
        'best_method': best_method,
        'method_scores': {m: r['avg_quality_score'] for m, r in results.items()},
        'recommendation': 'combined' if results['combined']['avg_quality_score'] > 0.1 else best_method
    }
    
    with open(output_dir / "preprocessing_evaluation.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_dir}")
    
    # 最適化された前処理クラスをエクスポート
    optimized_code = '''
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
'''
    
    with open(output_dir / "granulate_preprocessor.py", 'w') as f:
        f.write(optimized_code)
    
    print("\nOptimized preprocessor code saved to: preprocessing_results/granulate_preprocessor.py")


if __name__ == "__main__":
    main()