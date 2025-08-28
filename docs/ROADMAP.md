# Granulate OCR ロードマップ（現状サマリと次の提案）

## 現状サマリ（これまで）
- モデル: CRNN（CNN + BiLSTM + CTC）。文字集合は A–Z（26文字）を前提に学習済み。
- 精度/速度（報告値）: 文字レベル 86.1%、単語レベル 61.6%、平均処理 25ms。
- API: `POST /ocr/process`（multipart）、`/ocr/process-base64`（JSON）。
- 前処理: 反転判定、バイラテラル、CLAHE、二値化、パディング付きリサイズ（64×256）。
- 既存資料: `docs/progress_report.md`, `docs/technical_details.md`, `docs/api_documentation.md`。

## 直近で実装した改善（このブランチ）
- デバイス自動選択: `mps`/`cuda`/`cpu` の自動切替（環境変数で上書き可）。
- CTC ビーム探索の強化:
  - 簡易言語モデル（unigram/bigram JSON）による確率加点（`CRNN_LM_*`）。
  - 語彙制約（プレフィックス制約＋語彙ヒット時のボーナス）。
  - 任意の語彙オートコレクト（近似一致で終端補正）。
  - 誤認識傾向対策（E への寄り補正、R/D/N のブースト）。
- 文字集合の外部化: `CRNN_CHARSET` で将来の `0-9` 拡張に備えたコード整合。
- Transformer OCR のPoC骨組み: `scripts/transformer_ocr.py`（将来検証用）。

## すぐに試せる設定（推奨）
- 誤認識対策（まずは軽めに）
  - `CRNN_CONFUSION_TWEAKS=true`
  - `CRNN_E_PENALTY=0.95`（0.90〜0.98で探索）
  - `CRNN_RDN_BOOST=1.05`（1.02〜1.10で探索）
- 語彙とオートコレクト（実単語が多い場合）
  - `CRNN_LEXICON_PATH=/path/to/words.txt`
  - `CRNN_LEXICON_STRICT=true`（未知語を抑止したい時）
  - `CRNN_LEXICON_AUTOCORRECT=true`（近似一致で補正）
- 簡易LM（文字頻度/ビグラム）
  - `CRNN_LM_PATH=/path/to/lm.json`
  - `CRNN_LM_WEIGHT=0.2`（0.1〜0.3で探索）
- デバイス
  - `CRNN_DEVICE=mps|cuda|cpu`（未指定なら自動）

サンプル設定:
- 語彙: `configs/ocr/lexicon_words.txt`
- LM: `configs/ocr/char_lm.json`

## 環境変数（新規・拡張）
- デバイス/モデル
  - `CRNN_DEVICE` / `OCR_DEVICE` / `TORCH_DEVICE`
  - `CRNN_MODEL_PATH`, `CRNN_CHARSET`
- デコーダ/前処理
  - `CRNN_BEAM_SEARCH`, `CRNN_BEAM_WIDTH`
  - `CRNN_BEAM_LENGTH_PENALTY`
  - `CRNN_WIDTH_SCALE`, `CRNN_RIGHT_MARGIN`, `CRNN_MAX_WIDTH`
- 語彙/LM
  - `CRNN_LEXICON_PATH` or `CRNN_LEXICON`
  - `CRNN_LEXICON_STRICT`, `CRNN_LEXICON_BONUS`
  - `CRNN_LEXICON_AUTOCORRECT`, `CRNN_LEXICON_AUTOCORRECT_MIN_RATIO`
  - `CRNN_LM_PATH`, `CRNN_LM_WEIGHT`
- 混同補正
  - `CRNN_CONFUSION_TWEAKS`, `CRNN_E_PENALTY`, `CRNN_RDN_BOOST`

## 次の提案ステップ
1) 誤認識対策のパラメータチューニング（短期）
- 目的: R→E、D→E、N→E の抑制と全体精度の改善。
- 手順: グリッド探索（`E_PENALTY`×`RDN_BOOST`×`BEAM_WIDTH`×`LM_WEIGHT`）。
- 成果物: 推奨デフォルト値、テストレポートの更新。

2) 語彙とLMの準備（短期）
- `words.txt`（プロジェクト固有語彙）と `lm.json`（unigram/bigram）を作成。
- ユースケース別プリセット（一般英単語、専用用語）を用意。

3) 評価パイプラインの整備（短期）
- 既存テスト画像一括評価、文字/単語精度、編集距離、処理時間の自動集計。
- 誤りヒートマップ（混同行列）を出力。

 実行例（本リポジトリの評価スクリプト）:
 ```bash
 python scripts/evaluate_crnn_params.py \
   --data-dir test_data \
   --service fixed \
   --lexicon configs/ocr/lexicon_words.txt \
   --lm configs/ocr/char_lm.json \
   --limit 0
 ```

### パラメータ探索レポート（最新）
- データ: `test_data`（判定対象 87枚）
- サービス: `OCRServiceWithCRNNFixed`（CRNN優先、失敗時フォールバック）
- グリッド（16通り）: `E_PENALTY∈{0.90,0.95}`, `RDN_BOOST∈{1.00,1.05}`, `BEAM_WIDTH∈{3,5}`, `LM_WEIGHT∈{0.0,0.2}`
- 語彙/LM: `configs/ocr/lexicon_words.txt`, `configs/ocr/char_lm.json`

結果（要約）
- 語彙厳格ON（デフォルト）: ベスト構成は `{'CRNN_E_PENALTY': 0.9, 'CRNN_RDN_BOOST': 1.0, 'CRNN_BEAM_WIDTH': 3, 'CRNN_LM_WEIGHT': 0.0}`、単語精度 1.1%、文字精度 11.0%、平均 50ms。
- 語彙厳格OFF: ベスト構成は同一で、単語精度 1.1%、文字精度 11.2%、平均 37ms。
- 多くの画像で CRNN 出力が空→フォールバック（"CRNN failed...") が発生。語彙外の単語が多い/前処理ギャップが主因と推測。

推奨（短期）
- 運用では `CRNN_LEXICON_STRICT=false` を既定（未知語許容）。
- 語彙をテストデータから自動生成し、厳格ONでも試験（下記スクリプト）。
- CRNN前処理パラメータ（`CRNN_WIDTH_SCALE`, `CRNN_RIGHT_MARGIN`）も小規模グリッドで探索。

語彙の自動生成（ファイル名から抽出）
```bash
python scripts/generate_lexicon_from_dataset.py \
  --data-dir test_data \
  --glob "*.png" \
  --output configs/ocr/lexicon_from_dataset.txt

# 厳格ONで評価
CRNN_LEXICON_STRICT=true \
python scripts/evaluate_crnn_params.py \
  --data-dir test_data \
  --service fixed \
  --lexicon configs/ocr/lexicon_from_dataset.txt \
  --lm configs/ocr/char_lm.json
```

前処理グリッドの探索（横幅スケール/右マージン）
```bash
python scripts/evaluate_crnn_params.py \
  --data-dir test_data \
  --service fixed \
  --lexicon configs/ocr/lexicon_words.txt \
  --lm configs/ocr/char_lm.json \
  --ws-list 1.0 1.08 1.15 \
  --rm-list 12 24
```

追加評価（語彙自動生成＋前処理グリッド）
- 語彙: `configs/ocr/lexicon_from_dataset.txt`（自動生成、71語）＋厳格ON
- 前処理グリッド: `CRNN_WIDTH_SCALE∈{1.0, 1.08}`, `CRNN_RIGHT_MARGIN∈{12,24}`
- ベスト構成: `{'CRNN_E_PENALTY': 0.9, 'CRNN_RDN_BOOST': 1.0, 'CRNN_BEAM_WIDTH': 3, 'CRNN_LM_WEIGHT': 0.0, 'CRNN_WIDTH_SCALE': 1.08, 'CRNN_RIGHT_MARGIN': 12}`
- 精度: 単語 3.4%、文字 10.7%、平均 32.6ms（87枚）

所感:
- 語彙をデータセット準拠に拡充＋前処理微調整で、CRNNの空出力は一部改善し単語精度はわずかに改善。
- 一方で文字精度は大きくは伸びず、学習-実データの見た目ギャップがボトルネック。半教師ありや追加学習が有効。

4) 数字 0–9 サポート（中期）
- `CRNN_CHARSET=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789` で再学習。
- 新チェックポイント（出力37=36+blank）を `CRNN_MODEL_PATH` で切替。
- API/マッピング/テスト更新。

5) Transformer OCR のPoC（中期）
- 目的: CRNNとの比較検証（精度/速度/ロバスト性）。
- 小規模データで実験→有望なら本格統合（ビーム＋LM/辞書と連携）。

6) 半教師あり（中期）
- 擬似ラベル: 高信頼（例: ≥0.9）の推論結果のみ採用し再学習。
- データ選別とバランス管理、過学習チェック。

7) パフォーマンス/展開（横断）
- GPU活用・バッチ化、TorchScript/ONNX化、量子化（int8）実験。
- 前処理のSIMD最適化、I/Oの非同期化。

## マイルストーン例
- M1（1週）: 語彙/LM導入＋パラメータ最適化、評価自動化。
- M2（2週）: 数字拡張の再学習とデプロイ準備。
- M3（3週）: Transformer PoC 比較、半教師あり計画立案。

## 参照
- 進捗: `docs/progress_report.md`
- 技術詳細: `docs/technical_details.md`
- API: `docs/api_documentation.md`
- 次のステップ（従来版）: `docs/NEXT_STEPS.md`
