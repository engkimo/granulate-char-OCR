# Granulate OCR 技術詳細ドキュメント

## システムアーキテクチャ

### 全体構成

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │────▶│  OCR Service    │
│  (React/TS)     │     │   (FastAPI)     │     │  (Python/ML)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    │                                           │
                              ┌─────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
                              │   CRNN    │  │    CNN      │  │  Tesseract  │
                              │  (主要)   │  │ (文字単位)  │  │  (補助)     │
                              └───────────┘  └─────────────┘  └─────────────┘
```

### Clean Architecture 層構造

```
backend/
├── domain/           # ビジネスエンティティ（外部依存なし）
│   └── entities/
│       ├── ocr_result.py    # OCR結果エンティティ
│       └── character.py     # 文字エンティティ
│
├── application/      # ビジネスロジック
│   └── services/
│       ├── ocr_service.py           # 基本OCRサービス
│       └── ocr_service_crnn.py      # CRNN統合版
│
├── infrastructure/   # 外部システム連携
│   └── mapping/
│       └── granulate_mapping.py     # 文字マッピング
│
└── api/             # Web API層
    └── endpoints/
        └── ocr.py                   # OCRエンドポイント
```

## CRNNモデル詳細

### モデルアーキテクチャ

```python
CRNN(
  # CNN特徴抽出器
  cnn: Sequential(
    # Block 1
    Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
    ReLU
    MaxPool2d(2, 2)
    
    # Block 2
    Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
    ReLU
    MaxPool2d(2, 2)
    
    # Block 3
    Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
    BatchNorm2d(256)
    ReLU
    Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
    ReLU
    MaxPool2d((4, 1), (2, 1))
    
    # Block 4
    Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
    BatchNorm2d(512)
    ReLU
    Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
    ReLU
    MaxPool2d((4, 1), (2, 1))
    
    # Final
    Conv2d(512, 512, kernel_size=(2, 1))
  )
  
  # RNN層
  rnn: Sequential(
    BidirectionalLSTM(512, 256, 256)
    BidirectionalLSTM(256, 256, 27)  # 27 = 26文字 + blank
  )
)
```

### 入出力仕様

- **入力**: 64×256ピクセルのグレースケール画像
- **出力**: 可変長文字列（CTC デコーディング）

### CTCLabelConverter

```python
class CTCLabelConverter:
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def encode(self, text: str) -> List[int]:
        # テキストをインデックス列に変換
        # 'HELLO' → [7, 4, 11, 11, 14]
        
    def decode(self, indices: torch.Tensor) -> List[str]:
        # CTCデコーディング
        # 連続する同一文字を削除
        # blank文字を削除
```

## 画像前処理パイプライン

### 基本前処理フロー

```python
def preprocess_image(image):
    # 1. グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 背景判定と反転
    if np.mean(gray) > 128:  # 白背景
        gray = 255 - gray
    
    # 3. ノイズ除去
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 4. コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 5. 二値化
    _, binary = cv2.threshold(enhanced, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. リサイズ（アスペクト比保持）
    resized = resize_with_padding(binary, target_size=(64, 256))
    
    return resized
```

### 文字セグメンテーション（CNNモード用）

```python
def segment_characters(binary_image):
    # 水平投影プロファイル
    horizontal_projection = np.sum(binary_image, axis=0)
    
    # 文字境界検出
    boundaries = find_boundaries(horizontal_projection)
    
    # 各文字を切り出し
    characters = []
    for start, end in boundaries:
        char_img = binary_image[:, start:end]
        characters.append(char_img)
    
    return characters
```

## データ拡張戦略

### 訓練データ生成

1. **オリジナル文字生成**
   ```python
   # フォント: Goma Shin Mincho
   # サイズ: 100px
   # 色: 紫（#8B008B）
   # 背景: 透明
   ```

2. **太さバリエーション**
   ```python
   thickness_levels = [1, 3, 5, 7]  # ピクセル
   for thickness in thickness_levels:
       kernel = cv2.getStructuringElement(
           cv2.MORPH_ELLIPSE, (thickness, thickness)
       )
       thick_char = cv2.dilate(char_image, kernel)
   ```

3. **背景バリエーション**
   - 黒背景（#000000）
   - 白背景（#FFFFFF）
   - グラデーション（垂直/水平）
   - テクスチャ（ノイズ付き）

4. **ノイズとエフェクト**
   ```python
   # ガウシアンノイズ
   noise = np.random.normal(0, 10, image.shape)
   
   # モーションブラー
   kernel = np.zeros((15, 15))
   kernel[7, :] = np.ones(15) / 15
   blurred = cv2.filter2D(image, -1, kernel)
   ```

## API仕様

### エンドポイント

#### POST /api/v1/ocr/process-base64
Base64エンコードされた画像を処理

**リクエスト:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANS..."
}
```

**レスポンス:**
```json
{
  "image_id": "img_a1b2c3d4",
  "text": "PLEASURE",
  "average_confidence": 0.892,
  "processing_time": 0.0245,
  "characters": [
    {
      "granulate_symbol": "GP",
      "latin_equivalent": "P",
      "confidence": 0.95
    },
    ...
  ]
}
```

## パフォーマンス最適化

### 推論高速化

1. **バッチ正規化の無効化**
   ```python
   model.eval()  # BatchNormを推論モードに
   ```

2. **勾配計算の無効化**
   ```python
   with torch.no_grad():
       output = model(input)
   ```

3. **テンソルのメモリ配置**
   ```python
   tensor = tensor.contiguous()  # メモリ連続性の確保
   ```

### メモリ最適化

1. **画像サイズ制限**
   - 最大幅: 256ピクセル
   - 最大高さ: 64ピクセル

2. **モデルロード戦略**
   ```python
   # 必要時のみロード
   if self.crnn_model is None:
       self._load_crnn_model()
   ```

## トラブルシューティング

### よくある問題と解決策

1. **低精度の原因**
   - 訓練データとテストデータの特性不一致
   - 文字の太さの違い
   - 背景色の違い

2. **処理速度の問題**
   - CPUのみでの推論（GPU未使用）
   - 画像サイズが大きすぎる
   - 不要な前処理ステップ

3. **メモリエラー**
   - 大きすぎる画像
   - バッチサイズが大きすぎる
   - メモリリークの可能性

## 評価メトリクス

### 精度指標

1. **文字レベル精度**
   ```python
   correct_chars = sum(1 for e, r in zip(expected, recognized) if e == r)
   accuracy = correct_chars / total_chars
   ```

2. **単語レベル精度**
   ```python
   word_accuracy = 1.0 if recognized == expected else 0.0
   ```

3. **編集距離**
   ```python
   from Levenshtein import distance
   edit_dist = distance(expected, recognized)
   ```

### パフォーマンス指標

- 平均処理時間
- 95パーセンタイル処理時間
- スループット（画像/秒）

## 今後の拡張計画

1. **Transformer ベースモデル**
   - Vision Transformer の採用
   - 自己注意機構による精度向上

2. **半教師あり学習**
   - 大量の未ラベルデータの活用
   - 自己学習による性能向上

3. **マルチモーダル認識**
   - コンテキスト情報の活用
   - 周辺テキストとの関連性考慮