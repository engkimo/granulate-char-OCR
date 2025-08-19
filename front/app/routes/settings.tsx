import type { Route } from "./+types/settings"
import { Link } from 'react-router'
import { useState } from 'react'

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Settings - Granulate OCR" },
    { name: "description", content: "Configure OCR settings" },
  ]
}

export default function Settings() {
  const [minConfidence, setMinConfidence] = useState(0.5)
  const [apiUrl, setApiUrl] = useState(import.meta.env.VITE_API_URL || 'http://localhost:8000')
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    // In a real app, we'd persist these settings
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Settings
          </h1>
          <Link
            to="/"
            className="text-blue-600 hover:text-blue-700"
          >
            ← Back to Scanner
          </Link>
        </header>

        <div className="bg-white rounded-lg shadow p-6 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Minimum Confidence Threshold
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>0%</span>
              <span className="font-medium">{Math.round(minConfidence * 100)}%</span>
              <span>100%</span>
            </div>
            <p className="text-sm text-gray-600 mt-2">
              Characters with confidence below this threshold will be highlighted
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              API URL
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-sm text-gray-600 mt-1">
              Backend API endpoint for OCR processing
            </p>
          </div>

          <div className="pt-4 border-t">
            <button
              onClick={handleSave}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              Save Settings
            </button>
            {saved && (
              <p className="text-sm text-green-600 text-center mt-2">
                Settings saved successfully!
              </p>
            )}
          </div>
        </div>

        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">About</h2>
          <div className="space-y-2 text-sm text-gray-600">
            <p>Granulate Character OCR v0.1.0</p>
            <p>Recognition system for Kamen Rider Gavv Granulate characters</p>
            <p className="pt-2">
              <a 
                href="https://github.com/yourusername/granulate-char-ocr"
                className="text-blue-600 hover:text-blue-700"
                target="_blank"
                rel="noopener noreferrer"
              >
                View on GitHub
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}