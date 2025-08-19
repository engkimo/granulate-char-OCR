import { describe, it, expect, vi, beforeEach } from 'vitest'
import axios from 'axios'
import { ocrApi } from './ocr'

vi.mock('axios')

describe('OCR API Service', () => {
  const mockApiUrl = 'http://localhost:8000'
  const mockImageBlob = new Blob(['mock-image'], { type: 'image/png' })
  
  beforeEach(() => {
    vi.clearAllMocks()
    // Mock environment variable
    process.env.VITE_API_URL = mockApiUrl
  })

  describe('processImage', () => {
    it('should send image blob to API and return result', async () => {
      const mockResponse = {
        data: {
          image_id: 'test-123',
          text: 'ABC123',
          average_confidence: 0.92,
          processing_time: 0.256,
          timestamp: '2024-01-01T12:00:00Z',
          characters: [
            { granulate_symbol: 'ᐁ', latin_equivalent: 'A', confidence: 0.95 }
          ]
        }
      }
      
      vi.mocked(axios.post).mockResolvedValueOnce(mockResponse)
      
      const result = await ocrApi.processImage(mockImageBlob)
      
      expect(axios.post).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/ocr/process`,
        expect.any(FormData),
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )
      
      // Check FormData was constructed correctly
      const formDataCall = vi.mocked(axios.post).mock.calls[0][1]
      expect(formDataCall).toBeInstanceOf(FormData)
      
      expect(result).toEqual(mockResponse.data)
    })

    it('should handle API errors gracefully', async () => {
      const mockError = new Error('Network error')
      vi.mocked(axios.post).mockRejectedValueOnce(mockError)
      
      await expect(ocrApi.processImage(mockImageBlob)).rejects.toThrow('Network error')
    })

    it('should handle validation errors from API', async () => {
      const mockError = {
        response: {
          status: 400,
          data: { detail: 'Invalid file type' }
        }
      }
      vi.mocked(axios.post).mockRejectedValueOnce(mockError)
      
      await expect(ocrApi.processImage(mockImageBlob)).rejects.toEqual(mockError)
    })
  })

  describe('processImageBase64', () => {
    it('should send base64 image to API', async () => {
      const mockBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANS...'
      const mockResponse = {
        data: {
          image_id: 'test-456',
          text: 'XYZ',
          average_confidence: 0.88,
          processing_time: 0.150,
          timestamp: '2024-01-01T13:00:00Z',
          characters: []
        }
      }
      
      vi.mocked(axios.post).mockResolvedValueOnce(mockResponse)
      
      const result = await ocrApi.processImageBase64(mockBase64)
      
      expect(axios.post).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/ocr/process-base64`,
        { image: mockBase64.split(',')[1] }, // Remove data URL prefix
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )
      
      expect(result).toEqual(mockResponse.data)
    })

    it('should extract base64 data from data URL', async () => {
      const mockBase64WithPrefix = 'data:image/png;base64,ABC123=='
      const mockResponse = { data: { text: 'test' } }
      
      vi.mocked(axios.post).mockResolvedValueOnce(mockResponse)
      
      await ocrApi.processImageBase64(mockBase64WithPrefix)
      
      const callArgs = vi.mocked(axios.post).mock.calls[0]
      expect(callArgs[1]).toEqual({ image: 'ABC123==' })
    })
  })

  describe('getHealth', () => {
    it('should check API health status', async () => {
      const mockResponse = {
        data: {
          status: 'healthy',
          version: '0.1.0',
          timestamp: '2024-01-01T12:00:00Z'
        }
      }
      
      vi.mocked(axios.get).mockResolvedValueOnce(mockResponse)
      
      const result = await ocrApi.getHealth()
      
      expect(axios.get).toHaveBeenCalledWith(`${mockApiUrl}/api/v1/health`)
      expect(result).toEqual(mockResponse.data)
    })
  })
})