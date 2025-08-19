import { type OCRResponse } from '~/types/ocr'

interface OCRResultProps {
  result?: OCRResponse
}

export function OCRResult({ result }: OCRResultProps) {
  if (!result) {
    return (
      <div className="p-8 text-center text-gray-500">
        No result available
      </div>
    )
  }

  if (result.characters.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        No characters detected
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Main result text */}
      <div className="mb-6">
        <h2 className="text-sm font-medium text-gray-600 mb-2">Recognized Text</h2>
        <p className="text-3xl font-bold text-gray-900">{result.text}</p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <p className="text-sm text-gray-600">Average Confidence</p>
          <p className="text-lg font-semibold">
            {Math.round(result.average_confidence * 100)}%
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Processing Time</p>
          <p className="text-lg font-semibold">{result.processing_time}s</p>
        </div>
      </div>

      {/* Character breakdown */}
      <div>
        <h3 className="text-sm font-medium text-gray-600 mb-3">Character Details</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {result.characters.map((char, index) => (
            <div
              key={index}
              data-testid={`character-${index}`}
              className={`
                border rounded-lg p-3 text-center
                ${char.confidence < 0.5 ? 'low-confidence border-red-300 bg-red-50' : 'border-gray-200'}
              `}
            >
              <div className="text-2xl font-bold mb-1">{char.granulate_symbol}</div>
              <div className="text-sm text-gray-600">→</div>
              <div className="text-xl font-semibold">{char.latin_equivalent}</div>
              <div className="text-sm text-gray-500 mt-1">
                {Math.round(char.confidence * 100)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}