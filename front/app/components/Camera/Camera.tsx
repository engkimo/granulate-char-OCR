import { useEffect, useRef, useState } from 'react'

interface CameraProps {
  onCapture: (imageBlob: Blob) => void
}

type CameraFacing = 'user' | 'environment'

export function Camera({ onCapture }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [facing, setFacing] = useState<CameraFacing>('environment')

  useEffect(() => {
    startCamera()
    
    return () => {
      stopCamera()
    }
  }, [facing])

  const startCamera = async () => {
    try {
      setIsLoading(true)
      setError(null)
      
      // Stop existing stream if any
      stopCamera()
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing },
        audio: false
      })
      
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
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0)
    
    // Convert to blob
    canvas.toBlob((blob) => {
      if (blob) {
        onCapture(blob)
      }
    }, 'image/png')
  }

  const handleSwitchCamera = () => {
    setFacing(prev => prev === 'environment' ? 'user' : 'environment')
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
      
      <video
        ref={videoRef}
        data-testid="camera-video"
        className="w-full rounded-lg shadow-lg"
        playsInline
        muted
      />
      
      <canvas
        ref={canvasRef}
        className="hidden"
      />
      
      <div className="mt-4 flex gap-4 justify-center">
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
    </div>
  )
}