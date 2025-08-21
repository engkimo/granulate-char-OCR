import { useState } from 'react'
import type { Route } from "./+types/_index"
import { Camera } from '~/components/Camera'
import { OCRResult } from '~/components/OCRResult'
import { useOCRStore } from '~/stores/useOCRStore'
import { ocrApi } from '~/services/api'
import { useMutation } from '@tanstack/react-query'

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Granulate Character OCR" },
    { name: "description", content: "OCR system for Kamen Rider Gavv Granulate characters" },
  ]
}

export default function Index() {
  const [showCamera, setShowCamera] = useState(true)
  const { currentResult, isProcessing, error, setProcessing, setResult, setError } = useOCRStore()

  const processImageMutation = useMutation({
    mutationFn: ocrApi.processImage,
    onMutate: () => {
      setProcessing(true)
      setShowCamera(false)
    },
    onSuccess: (data) => {
      setResult(data)
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : 'Failed to process image')
      setShowCamera(true)
    }
  })

  const handleCapture = (imageBlob: Blob, previewUrl: string) => {
    processImageMutation.mutate(imageBlob)
  }

  const handleNewCapture = () => {
    setShowCamera(true)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Granulate Character OCR
          </h1>
          <p className="text-gray-600">
            Scan and translate Kamen Rider Gavv Granulate characters
          </p>
        </header>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {showCamera && !isProcessing ? (
          <Camera onCapture={handleCapture} />
        ) : isProcessing ? (
          <div className="flex flex-col items-center justify-center p-16 bg-white rounded-lg shadow-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mb-4"></div>
            <p className="text-gray-600">Processing image...</p>
          </div>
        ) : currentResult ? (
          <div>
            <OCRResult result={currentResult} />
            <div className="mt-6 text-center">
              <button
                onClick={handleNewCapture}
                className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              >
                Scan Another
              </button>
            </div>
          </div>
        ) : null}

        <div className="mt-12 grid grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="font-semibold text-gray-900 mb-2">How it works</h2>
            <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
              <li>Point camera at Granulate text</li>
              <li>Capture the image</li>
              <li>Get instant translation</li>
            </ol>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="font-semibold text-gray-900 mb-2">Tips</h2>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
              <li>Use good lighting</li>
              <li>Keep text in focus</li>
              <li>Avoid glare</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}