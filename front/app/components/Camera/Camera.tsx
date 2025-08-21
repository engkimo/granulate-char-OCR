import { useEffect, useRef, useState } from 'react'

interface CameraProps {
  onCapture: (imageBlob: Blob, previewUrl: string) => void
}

type CameraFacing = 'user' | 'environment'
type ProcessingMode = 'none' | 'basic' | 'enhanced' | 'aggressive'

export function Camera({ onCapture }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const previewCanvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [facing, setFacing] = useState<CameraFacing>('environment')
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('aggressive')
  const [capturedImages, setCapturedImages] = useState<Array<{ url: string, timestamp: Date }>>([])

  useEffect(() => {
    startCamera()
    
    return () => {
      stopCamera()
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [facing])

  useEffect(() => {
    if (videoRef.current && previewCanvasRef.current && !isLoading) {
      drawPreview()
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isLoading, processingMode])

  const startCamera = async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      // Stop existing stream if any
      stopCamera()
      
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: facing,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          // 高度なカメラ設定
          // @ts-ignore - 型定義が不完全なため
          advanced: [{
            // ISO感度を下げてノイズを減少
            iso: { ideal: 100 },
            // 露出補正
            exposureCompensation: { ideal: 0 },
            // ホワイトバランス
            whiteBalanceMode: 'continuous',
            // フォーカスモード
            focusMode: 'continuous',
            // 明るさ
            brightness: { ideal: 0 },
            // コントラスト
            contrast: { ideal: 100 },
            // 彩度
            saturation: { ideal: 100 },
            // シャープネス
            sharpness: { ideal: 100 }
          }]
        },
        audio: false
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      streamRef.current = stream
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      
      setIsLoading(false)
    } catch (err) {
      setError('Camera access denied. Please enable camera permissions.')
      setIsLoading(false)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
  }

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return
    
    const video = videoRef.current
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    
    if (!context) return
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    
    // Save context state
    context.save()
    
    // If using front camera (user facing), flip the image horizontally
    if (facing === 'user') {
      context.scale(-1, 1)
      context.drawImage(video, -canvas.width, 0)
    } else {
      // Draw video frame to canvas normally for rear camera
      context.drawImage(video, 0, 0)
    }
    
    // Restore context state
    context.restore()
    
    // Apply image processing based on selected mode
    if (processingMode !== 'none') {
      applyImageProcessing(context, canvas.width, canvas.height, processingMode)
    }
    
    // Get preview URL before converting to blob
    const previewUrl = canvas.toDataURL('image/png')
    
    // Add to captured images history
    setCapturedImages(prev => [{
      url: previewUrl,
      timestamp: new Date()
    }, ...prev].slice(0, 5)) // Keep last 5 images
    
    // Convert to blob
    canvas.toBlob((blob) => {
      if (blob) {
        onCapture(blob, previewUrl)
      }
    }, 'image/png')
  }

  const applyImageProcessing = (
    context: CanvasRenderingContext2D, 
    width: number, 
    height: number, 
    mode: ProcessingMode
  ) => {
    const imageData = context.getImageData(0, 0, width, height)
    const data = imageData.data
    const processedData = new Uint8ClampedArray(data)
    
    switch (mode) {
      case 'basic':
        // Simple contrast enhancement
        for (let i = 0; i < processedData.length; i += 4) {
          for (let c = 0; c < 3; c++) {
            let value = processedData[i + c]
            value = Math.max(0, Math.min(255, (value - 128) * 1.1 + 128))
            processedData[i + c] = value
          }
        }
        break
        
      case 'enhanced':
        // Median filter for noise reduction
        for (let y = 1; y < height - 1; y++) {
          for (let x = 1; x < width - 1; x++) {
            for (let c = 0; c < 3; c++) {
              const neighbors = []
              for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                  const idx = ((y + dy) * width + (x + dx)) * 4 + c
                  neighbors.push(data[idx])
                }
              }
              neighbors.sort((a, b) => a - b)
              const median = neighbors[Math.floor(neighbors.length / 2)]
              const idx = (y * width + x) * 4 + c
              processedData[idx] = median
            }
          }
        }
        
        // Apply contrast enhancement
        for (let i = 0; i < processedData.length; i += 4) {
          for (let c = 0; c < 3; c++) {
            let value = processedData[i + c]
            value = Math.max(0, Math.min(255, (value - 128) * 1.2 + 128))
            processedData[i + c] = value
          }
        }
        break
        
      case 'aggressive':
        // zutomayo_OCRと同じ二値化処理を実装
        for (let i = 0; i < data.length; i += 4) {
          // RGBの平均値を計算
          const ave = (data[i] + data[i + 1] + data[i + 2]) / 3
          
          // 閾値128で二値化（白か黒か）
          const binaryValue = ave > 128 ? 255 : 0
          
          processedData[i] = binaryValue     // R
          processedData[i + 1] = binaryValue // G
          processedData[i + 2] = binaryValue // B
          processedData[i + 3] = 255         // Alpha
        }
        break
    }
    
    imageData.data.set(processedData)
    context.putImageData(imageData, 0, 0)
  }

  const handleSwitchCamera = () => {
    setFacing(prev => prev === 'environment' ? 'user' : 'environment')
  }

  const drawPreview = () => {
    if (!videoRef.current || !previewCanvasRef.current) return
    
    const video = videoRef.current
    const canvas = previewCanvasRef.current
    const context = canvas.getContext('2d')
    
    if (!context || !video.videoWidth || !video.videoHeight) {
      animationFrameRef.current = requestAnimationFrame(drawPreview)
      return
    }
    
    // Set canvas dimensions
    canvas.width = 300
    canvas.height = 100
    
    // Draw cropped region (similar to zutomayo_OCR)
    context.save()
    
    if (facing === 'user') {
      context.scale(-1, 1)
      context.drawImage(video, 100, 300, 600, 200, -300, 0, 300, 100)
    } else {
      context.drawImage(video, 100, 300, 600, 200, 0, 0, 300, 100)
    }
    
    context.restore()
    
    // Apply processing if needed
    if (processingMode !== 'none') {
      applyImageProcessing(context, canvas.width, canvas.height, processingMode)
    }
    
    animationFrameRef.current = requestAnimationFrame(drawPreview)
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-red-50 rounded-lg">
        <p className="text-red-600 text-center mb-4">{error}</p>
        <button
          onClick={startCamera}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="relative">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 rounded-lg">
          <p className="text-gray-600">Accessing camera...</p>
        </div>
      )}
      
      <div className="space-y-4">
        {/* Original video (hidden small) */}
        <video
          ref={videoRef}
          data-testid="camera-video"
          className="w-1 h-1 opacity-0 absolute"
          playsInline
          muted
        />
        
        {/* Preview canvas (main display) */}
        <canvas
          ref={previewCanvasRef}
          className="w-full max-w-[600px] mx-auto rounded-lg shadow-lg border-2 border-gray-300"
          width="300"
          height="100"
        />
        
        {/* Hidden canvas for capture */}
        <canvas
          ref={canvasRef}
          className="hidden"
        />
      </div>
      
      <div className="mt-4 space-y-4">
        <div className="flex gap-4 justify-center">
          <button
            onClick={handleCapture}
            disabled={isLoading}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Capture
          </button>
          
          <button
            onClick={handleSwitchCamera}
            disabled={isLoading}
            className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Switch Camera
          </button>
        </div>
        
        <div className="flex flex-col items-center gap-2">
          <label className="text-sm font-medium text-gray-700">
            画像処理モード:
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => setProcessingMode('none')}
              className={`px-3 py-1 rounded ${
                processingMode === 'none' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              なし
            </button>
            <button
              onClick={() => setProcessingMode('basic')}
              className={`px-3 py-1 rounded ${
                processingMode === 'basic' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              基本
            </button>
            <button
              onClick={() => setProcessingMode('enhanced')}
              className={`px-3 py-1 rounded ${
                processingMode === 'enhanced' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              強化
            </button>
            <button
              onClick={() => setProcessingMode('aggressive')}
              className={`px-3 py-1 rounded ${
                processingMode === 'aggressive' 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              高強度
            </button>
          </div>
          <p className="text-xs text-gray-500 text-center mt-1">
            {processingMode === 'none' && 'オリジナル画像をそのまま使用'}
            {processingMode === 'basic' && 'コントラスト調整のみ'}
            {processingMode === 'enhanced' && 'ノイズ除去＋コントラスト調整'}
            {processingMode === 'aggressive' && '強力な二値化（白黒のみ）- グラニュート文字に最適'}
          </p>
        </div>
      </div>
      
      {/* Captured images preview */}
      {capturedImages.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-medium text-gray-700 mb-2">撮影した画像:</h3>
          <div className="flex gap-2 overflow-x-auto">
            {capturedImages.map((image, index) => (
              <div key={index} className="flex-shrink-0">
                <img 
                  src={image.url} 
                  alt={`Captured ${index + 1}`}
                  className="w-32 h-32 object-cover rounded border-2 border-gray-300"
                />
                <p className="text-xs text-gray-500 mt-1">
                  {image.timestamp.toLocaleTimeString()}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}