# Granulate OCR API ドキュメント

## 概要

Granulate OCR APIは、仮面ライダーガヴのGranulate文字を認識し、対応するラテン文字（A-Z）に変換するRESTful APIです。

## ベースURL

```
http://localhost:8000/api/v1
```

## 認証

現在のバージョンでは認証は不要です。

## エンドポイント

### 1. ヘルスチェック

APIの稼働状態を確認します。

```http
GET /health
```

**レスポンス例:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-20T10:30:00Z"
}
```

### 2. 画像ファイルアップロード

画像ファイルを直接アップロードしてOCR処理を実行します。

```http
POST /ocr/process
Content-Type: multipart/form-data
```

**パラメータ:**
- `file` (required): 画像ファイル（PNG, JPEG, GIF対応）

**cURLの例:**
```bash
curl -X POST \
  http://localhost:8000/api/v1/ocr/process \
  -F "file=@/path/to/image.png"
```

**レスポンス例:**
```json
{
  "image_id": "img_a1b2c3d4",
  "text": "HELLO",
  "average_confidence": 0.923,
  "processing_time": 0.0156,
  "characters": [
    {
      "granulate_symbol": "GH",
      "latin_equivalent": "H",
      "confidence": 0.95
    },
    {
      "granulate_symbol": "GE",
      "latin_equivalent": "E",
      "confidence": 0.88
    },
    {
      "granulate_symbol": "GL",
      "latin_equivalent": "L",
      "confidence": 0.92
    },
    {
      "granulate_symbol": "GL",
      "latin_equivalent": "L",
      "confidence": 0.93
    },
    {
      "granulate_symbol": "GO",
      "latin_equivalent": "O",
      "confidence": 0.94
    }
  ]
}
```

### 3. Base64エンコード画像

Base64形式でエンコードされた画像を送信してOCR処理を実行します。

```http
POST /ocr/process-base64
Content-Type: application/json
```

**リクエストボディ:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAUA..."
}
```

**JavaScriptの例:**
```javascript
// 画像をBase64にエンコード
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      // データURLからBase64部分を抽出
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
  });
};

// APIリクエスト
const processImage = async (imageFile) => {
  const base64Image = await fileToBase64(imageFile);
  
  const response = await fetch('http://localhost:8000/api/v1/ocr/process-base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64Image
    })
  });
  
  return response.json();
};
```

**Pythonの例:**
```python
import requests
import base64

# 画像をBase64エンコード
with open('image.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# APIリクエスト
response = requests.post(
    'http://localhost:8000/api/v1/ocr/process-base64',
    json={'image': image_base64}
)

result = response.json()
print(f"認識結果: {result['text']}")
print(f"信頼度: {result['average_confidence']:.2%}")
```

## レスポンスフィールド

### OCRResponse

| フィールド | 型 | 説明 |
|------------|------|------|
| `image_id` | string | 処理された画像の一意識別子 |
| `text` | string | 認識されたテキスト全体 |
| `average_confidence` | float | 平均信頼度スコア (0.0-1.0) |
| `processing_time` | float | 処理時間（秒） |
| `characters` | array | 各文字の詳細情報 |

### CharacterResponse

| フィールド | 型 | 説明 |
|------------|------|------|
| `granulate_symbol` | string | Granulate文字の表現（例: "GA"） |
| `latin_equivalent` | string | 対応するラテン文字（A-Z） |
| `confidence` | float | その文字の信頼度スコア (0.0-1.0) |

## エラーレスポンス

### 400 Bad Request

無効なリクエストの場合に返されます。

```json
{
  "detail": "Invalid file type: text/plain. Only images are allowed."
}
```

### 500 Internal Server Error

サーバー側のエラーが発生した場合に返されます。

```json
{
  "detail": "Error processing image: Unable to decode image data"
}
```

## 使用上の注意

### 画像の要件

1. **対応フォーマット**: PNG, JPEG, GIF
2. **推奨サイズ**: 幅 256ピクセル以下
3. **文字の向き**: 水平（横書き）
4. **背景**: 単色背景推奨

### パフォーマンス

- 平均処理時間: 25ms/画像
- 最大画像サイズ: 10MB
- 同時リクエスト数: 制限なし（サーバーリソースに依存）

### 精度向上のヒント

1. **高コントラスト**: 文字と背景のコントラストが高い画像
2. **適切な解像度**: 文字が明瞭に見える程度の解像度
3. **ノイズの少ない画像**: ぼやけやノイズが少ない画像
4. **正しい向き**: 文字が正立している画像

## 環境変数

APIの動作は以下の環境変数で制御できます：

| 変数名 | デフォルト | 説明 |
|--------|------------|------|
| `USE_CRNN` | `true` | CRNNモデルの使用有無 |
| `LOG_LEVEL` | `INFO` | ログレベル |
| `MAX_IMAGE_SIZE` | `10485760` | 最大画像サイズ（バイト） |

## サンプルアプリケーション

### シンプルなHTML/JavaScriptクライアント

```html
<!DOCTYPE html>
<html>
<head>
    <title>Granulate OCR Demo</title>
</head>
<body>
    <h1>Granulate OCR Demo</h1>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="processImage()">認識</button>
    <div id="result"></div>

    <script>
        async function processImage() {
            const input = document.getElementById('imageInput');
            const file = input.files[0];
            
            if (!file) {
                alert('画像を選択してください');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('http://localhost:8000/api/v1/ocr/process', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                document.getElementById('result').innerHTML = `
                    <h2>結果</h2>
                    <p>テキスト: ${result.text}</p>
                    <p>信頼度: ${(result.average_confidence * 100).toFixed(1)}%</p>
                    <p>処理時間: ${(result.processing_time * 1000).toFixed(1)}ms</p>
                `;
            } catch (error) {
                alert('エラー: ' + error.message);
            }
        }
    </script>
</body>
</html>
```

## トラブルシューティング

### Q: 「API server is not responding」エラーが出ます

A: APIサーバーが起動していることを確認してください：
```bash
uv run uvicorn backend.main:app --reload
```

### Q: 認識精度が低い

A: 以下を確認してください：
1. 画像の品質（解像度、コントラスト）
2. 文字の向き（水平であること）
3. 背景ノイズの有無

### Q: 処理が遅い

A: 以下の対策を検討してください：
1. 画像サイズを小さくする（最大256px幅推奨）
2. GPUの使用を検討する
3. バッチ処理の実装

## 更新履歴

### v1.1.0 (2025-01-20)
- CRNNモデルの統合
- 精度の大幅向上（86.1%文字レベル精度）

### v1.0.0 (2025-01-15)
- 初回リリース
- CNN + Tesseract + ハッシュベース認識