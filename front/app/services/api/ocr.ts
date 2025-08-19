import axios from 'axios'
import { type OCRResponse } from '~/types/ocr'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const ocrApi = {
  async processImage(imageBlob: Blob): Promise<OCRResponse> {
    const formData = new FormData()
    formData.append('file', imageBlob, 'capture.png')
    
    const response = await axios.post<OCRResponse>(
      `${API_URL}/api/v1/ocr/process`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    )
    
    return response.data
  },

  async processImageBase64(base64Image: string): Promise<OCRResponse> {
    // Remove data URL prefix if present
    const base64Data = base64Image.includes(',') 
      ? base64Image.split(',')[1] 
      : base64Image
    
    const response = await axios.post<OCRResponse>(
      `${API_URL}/api/v1/ocr/process-base64`,
      { image: base64Data },
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    )
    
    return response.data
  },

  async getHealth() {
    const response = await axios.get(`${API_URL}/api/v1/health`)
    return response.data
  }
}