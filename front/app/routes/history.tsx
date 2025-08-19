import type { Route } from "./+types/history"
import { useOCRStore } from '~/stores/useOCRStore'
import { OCRResult } from '~/components/OCRResult'
import { Link } from 'react-router'

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Recognition History - Granulate OCR" },
    { name: "description", content: "View past OCR recognition results" },
  ]
}

export default function History() {
  const { history, clearHistory } = useOCRStore()

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Recognition History
            </h1>
            <p className="text-gray-600">
              Your past {history.length} recognition{history.length !== 1 ? 's' : ''}
            </p>
          </div>
          <div className="flex gap-4">
            <Link
              to="/"
              className="px-4 py-2 text-blue-600 hover:text-blue-700"
            >
              Back to Scanner
            </Link>
            {history.length > 0 && (
              <button
                onClick={clearHistory}
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
              >
                Clear History
              </button>
            )}
          </div>
        </header>

        {history.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <p className="text-gray-500 mb-4">No recognition history yet</p>
            <Link
              to="/"
              className="text-blue-600 hover:text-blue-700"
            >
              Start scanning
            </Link>
          </div>
        ) : (
          <div className="space-y-6">
            {history.map((result, index) => (
              <div key={`${result.image_id}-${index}`}>
                <div className="text-sm text-gray-500 mb-2">
                  {new Date(result.timestamp).toLocaleString()}
                </div>
                <OCRResult result={result} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}