import { useState } from 'react'
import type { Route } from "./+types/_index"
import { Camera } from '~/components/Camera'
import { OCRResult } from '~/components/OCRResult'
import { useOCRStore } from '~/stores/useOCRStore'
import { ocrApi } from '~/services/api'
import { useMutation } from '@tanstack/react-query'

export function meta({}: Route.MetaArgs) {
  return [
    { title: "ストマック家 グラニュートOCR" },
    { name: "description", content: "仮面ライダーガヴ ストマック家のグラニュート文字解析システム" },
  ]
}

export default function Index() {
  const [showCamera, setShowCamera] = useState(true)
  const [capturedImageUrl, setCapturedImageUrl] = useState<string | null>(null)
  const { currentResult, isProcessing, error, setProcessing, setResult, setError } = useOCRStore()

  const processImageMutation = useMutation({
    mutationFn: ocrApi.processImage,
    onMutate: () => {
      setProcessing(true)
      setShowCamera(false)
    },
    onSuccess: (data) => {
      setResult({ ...data, imageUrl: capturedImageUrl || undefined })
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : 'Failed to process image')
      setShowCamera(true)
    }
  })

  const handleCapture = (imageBlob: Blob, previewUrl: string) => {
    setCapturedImageUrl(previewUrl)
    processImageMutation.mutate(imageBlob)
  }

  const handleNewCapture = () => {
    setShowCamera(true)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-purple-700 to-indigo-900">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-300 to-pink-300 mb-2 tracking-wide">
            ストマック家 グラニュート解析システム
          </h1>
          <p className="text-purple-200 text-lg">
            グラニュート文字を瞬時に解読
          </p>
        </header>

        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg backdrop-blur-sm">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {showCamera && !isProcessing ? (
          <Camera onCapture={handleCapture} />
        ) : isProcessing ? (
          <div className="flex flex-col items-center justify-center p-16 bg-purple-800/30 backdrop-blur-md rounded-lg shadow-2xl border border-purple-500/30">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-400 border-t-transparent mb-4"></div>
            <p className="text-purple-200">グラニュート文字を解析中...</p>
          </div>
        ) : currentResult ? (
          <div>
            <OCRResult result={currentResult} />
            <div className="mt-6 text-center">
              <button
                onClick={handleNewCapture}
                className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 shadow-lg transform hover:scale-105 transition-all duration-200"
              >
                もう一度スキャン
              </button>
            </div>
          </div>
        ) : null}

      </div>
    </div>
  )
}