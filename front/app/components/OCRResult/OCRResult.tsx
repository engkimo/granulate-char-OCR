import { type OCRResponse } from '~/types/ocr'

interface OCRResultProps {
  result?: OCRResponse
}

export function OCRResult({ result }: OCRResultProps) {
  if (!result) {
    return (
      <div className="p-8 text-center text-purple-300">
        結果がありません
      </div>
    )
  }

  if (result.characters.length === 0) {
    return (
      <div className="p-8 text-center text-purple-300">
        文字が検出されませんでした
      </div>
    )
  }

  return (
    <div className="bg-purple-900/30 backdrop-blur-md rounded-lg shadow-2xl p-6 border border-purple-500/30">
      {/* Captured image */}
      {result.imageUrl && (
        <div className="mb-6">
          <h2 className="text-sm font-medium text-purple-300 mb-2">キャプチャ画像</h2>
          <img 
            src={result.imageUrl} 
            alt="Captured for OCR" 
            className="w-full max-w-md mx-auto rounded-lg border-2 border-purple-500/50 shadow-lg"
          />
        </div>
      )}

      {/* Main result text */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-purple-300 mb-2">解読結果</h2>
        <p className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-300 to-pink-300">{result.text}</p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <p className="text-sm text-purple-300">平均信頼度</p>
          <p className="text-lg font-semibold text-purple-100">
            {Math.round(result.average_confidence * 100)}%
          </p>
        </div>
        <div>
          <p className="text-sm text-purple-300">処理時間</p>
          <p className="text-lg font-semibold text-purple-100">{result.processing_time}s</p>
        </div>
      </div>

      {/* Character breakdown */}
      <div>
        <h3 className="text-sm font-medium text-purple-300 mb-3">文字詳細</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {result.characters.map((char, index) => (
            <div
              key={index}
              data-testid={`character-${index}`}
              className={`
                border-2 rounded-lg p-3 text-center backdrop-blur-sm
                ${char.confidence < 0.5 ? 'low-confidence border-red-400/50 bg-red-900/20' : 'border-purple-500/50 bg-purple-800/20'}
              `}
            >
              <div className="text-2xl font-bold mb-1 text-purple-200">{char.granulate_symbol}</div>
              <div className="text-sm text-purple-400">→</div>
              <div className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-purple-300 to-pink-300">{char.latin_equivalent}</div>
              <div className="text-sm text-purple-400 mt-1">
                {Math.round(char.confidence * 100)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}