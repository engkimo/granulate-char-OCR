import { type OCRResponse } from '~/types/ocr'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const ocrApi = {
  async processImage(imageBlob: Blob): Promise<OCRResponse> {
    const formData = new FormData()
    formData.append('file', imageBlob, 'capture.png')
    
    const response = await fetch(`${API_URL}/api/v1/ocr/process`, {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return response.json()
  },

  async processImageBase64(base64Image: string): Promise<OCRResponse> {
    // Remove data URL prefix if present
    const base64Data = base64Image.includes(',') 
      ? base64Image.split(',')[1] 
      : base64Image
    
    const response = await fetch(`${API_URL}/api/v1/ocr/process-base64`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ image: base64Data })
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return response.json()
  },

  async getHealth() {
    const response = await fetch(`${API_URL}/api/v1/health`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    return response.json()
  }
}