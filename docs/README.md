# Granulate OCR Documentation

このディレクトリには、Granulate OCRシステムに関するドキュメントが含まれています。

## ドキュメント一覧

### 1. [進捗レポート](./progress_report.md)
プロジェクトの改善履歴と成果をまとめたドキュメント。初期状態から現在までの改善過程を時系列で記載。

### 2. [技術詳細](./technical_details.md)
システムアーキテクチャ、モデル構造、アルゴリズムの詳細な技術仕様。

### 3. [API仕様書](./api_documentation.md)
RESTful APIのエンドポイント、リクエスト/レスポンス形式、使用例を含む完全なAPI仕様。

## クイックリンク

- **プロジェクトルート**: [../README.md](../README.md)
- **CLAUDE.md**: [../CLAUDE.md](../CLAUDE.md) - 開発コマンドとガイドライン
- **テストデータ**: [../test_data/](../test_data/) - 評価用画像データ
- **モデル**: [../models/](../models/) - 訓練済みモデルファイル

## 主要な成果

- **初期精度**: 12.5% (1/8文字)
- **最終精度**: 86.1% (297/345文字)
- **改善率**: 6.9倍
- **処理速度**: 8倍高速化（200ms → 25ms）

## 技術スタック

- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: React, TypeScript, Vite
- **ML Models**: CRNN (CNN + Bidirectional LSTM + CTC)
- **Infrastructure**: Docker対応, Clean Architecture

## お問い合わせ

プロジェクトに関する質問や提案は、GitHubのIssueでお願いします。