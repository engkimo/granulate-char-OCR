# jTessBoxEditorでGranulate文字の.traineddataを作成する詳細手順

## 前提条件

- jTessBoxEditorが起動済み
- 学習用ファイルが準備済み（`training_data/tesseract/`ディレクトリ内）

## 手順

### 1. Box Editorタブで文字位置を確認・修正

#### 1.1 ファイルを開く

1. jTessBoxEditorの**「Box Editor」タブ**をクリック
2. メニューから**「File」→「Open」**を選択
3. ファイル選択ダイアログで以下のファイルを選択：
   ```
   /Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/gran.font.exp0.tif
   ```
4. 「Open」をクリック

#### 1.2 BOXファイルの確認

- 自動的に同じディレクトリの`gran.font.exp0.box`が読み込まれる
- 左側にページナビゲーション、中央に画像、右側に文字情報が表示される

#### 1.3 各ページの確認（重要）

1. **ページナビゲーション**で各ページを確認
2. 現在の問題：多くのページが黒背景のため文字が見えない可能性
3. 見えるページを探す：
   - Page 7, 27, 41, 42, 51, 56, 57 など（部分的に認識されたページ）
   - これらのページで文字と座標が正しいか確認

#### 1.4 必要に応じて修正

- 文字が間違っている場合：右側のテキストボックスで修正
- 座標が間違っている場合：
  1. 画像上で赤い枠をドラッグして調整
  2. または右側の数値を直接編集
- 修正後は**「Save」**ボタンで保存

### 2. Trainerタブで学習を実行

#### 2.1 Trainerタブに切り替え

1. **「Trainer」タブ**をクリック
2. 学習設定画面が表示される

#### 2.2 学習設定

以下の設定を行う：

| 項目 | 設定値 | 説明 |
|------|--------|------|
| **Tesseract Executable** | `/opt/homebrew/bin/tesseract` | Tesseractのパス（自動検出される場合あり） |
| **Training Data Directory** | `/Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/` | 学習データのディレクトリ |
| **Language** | `gran` | 作成する言語名 |
| **Bootstrap Language** | `eng` | ベースとする言語（英語） |
| **Training Mode** | `Train with Existing Box Files` | 既存のBOXファイルを使用 |
| **Font Name** | `font` | フォント名（gran.font.exp0の"font"部分） |

#### 2.3 ファイルの選択

1. **「TIFF/Box Pairs」**セクションで：
   - 「Add」ボタンをクリック
   - `gran.font.exp0.tif`を選択（BOXファイルは自動的にペアリング）
   
2. または**「Training Data」**フィールドに直接パスを入力：
   ```
   /Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/gran.font.exp0.tif
   ```

#### 2.4 追加オプション（任意）

- **「RTL」**（Right-to-Left）: チェックなし（左から右に読む文字）
- **「Dict Data」**: 辞書データがある場合は指定（オプション）

### 3. 学習の実行

#### 3.1 Trainボタンをクリック

1. 設定が完了したら**「Train」ボタン**をクリック
2. 進行状況がコンソールウィンドウに表示される

#### 3.2 学習プロセス

以下の処理が自動的に実行される：

```
1. Box Training (tesseract ... box.train)
2. Unicharset Extraction
3. Shape Clustering
4. MF Training
5. CN Training
6. Dictionary Creation
7. Combine Tessdata
```

#### 3.3 エラーの対処

もし「Empty page」エラーが多発する場合：

1. **「Run」メニュー**から**「Generate TIFF/Box Pair for Training」**を選択
2. 新しい白背景・黒文字の画像を生成
3. 再度学習を実行

### 4. 結果の確認

#### 4.1 生成されたファイル

学習が成功すると、以下のファイルが生成される：
```
/Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/gran.traineddata
```

#### 4.2 ファイルサイズの確認

```bash
ls -lh gran.traineddata
```
- 正常な場合：数KB〜数MB
- 異常な場合：0バイトまたは非常に小さい

### 5. インストールと動作確認

#### 5.1 Tessdataディレクトリにコピー

```bash
# Tessdataディレクトリを確認
tesseract --list-langs

# traineddataをコピー
sudo cp /Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/tesseract/gran.traineddata /opt/homebrew/share/tessdata/

# 確認
tesseract --list-langs | grep gran
```

#### 5.2 テスト実行

```bash
# テスト画像でOCRを実行
cd /Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR
tesseract training_data/extracted/A/A_00.png output -l gran --psm 10

# 結果を確認
cat output.txt
```

## トラブルシューティング

### 問題1: 多くのページが「Empty page」になる

**原因**: 画像が黒背景に白文字のため

**解決策**:
1. jTessBoxEditorの**「Tools」→「Merge TIFF」**で画像を前処理
2. または、少数の認識できたページだけで学習を続行

### 問題2: 学習が失敗する

**原因**: 必要なファイルが不足

**確認事項**:
- `unicharset`ファイルが存在するか
- `font_properties`ファイルが存在するか
- BOXファイルの形式が正しいか

### 問題3: 認識精度が低い

**改善策**:
1. より多くの学習サンプルを追加
2. 画像の品質を改善（コントラスト調整など）
3. 異なるPSMモードでテスト：
   ```bash
   # PSM 8: 単一単語として扱う
   tesseract image.png output -l gran --psm 8
   
   # PSM 10: 単一文字として扱う
   tesseract image.png output -l gran --psm 10
   ```

## 期待される結果

- 学習データが少ないため、初期精度は30-50%程度
- Hash-based認識（28.5%）より若干良い程度
- より良い結果にはCNN model（95%精度）の使用を推奨

## 次のステップ

1. 生成された`gran.traineddata`をOCRServiceに統合
2. 実際の画像でテストして精度を確認
3. 必要に応じて追加の学習データを作成

## 参考リンク

- [jTessBoxEditor公式ドキュメント](https://github.com/nguyenq/jTessBoxEditor/wiki)
- [Tesseract Training Wiki](https://tesseract-ocr.github.io/tessdoc/Training-Tesseract.html)