import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ocrApi } from './ocr'

// Mock fetch globally
global.fetch = vi.fn()

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
      const mockResponseData = {
        image_id: 'test-123',
        text: 'ABC123',
        average_confidence: 0.92,
        processing_time: 0.256,
        timestamp: '2024-01-01T12:00:00Z',
        characters: [
          { granulate_symbol: 'ᐁ', latin_equivalent: 'A', confidence: 0.95 }
        ]
      }
      
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponseData
      } as Response)
      
      const result = await ocrApi.processImage(mockImageBlob)
      
      expect(global.fetch).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/ocr/process`,
        {
          method: 'POST',
          body: expect.any(FormData)
        }
      )
      
      // Check FormData was constructed correctly
      const fetchCall = vi.mocked(global.fetch).mock.calls[0]
      expect(fetchCall[1]?.body).toBeInstanceOf(FormData)
      
      expect(result).toEqual(mockResponseData)
    })

    it('should handle API errors gracefully', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 500
      } as Response)
      
      await expect(ocrApi.processImage(mockImageBlob)).rejects.toThrow('HTTP error! status: 500')
    })

    it('should handle validation errors from API', async () => {
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: false,
        status: 400
      } as Response)
      
      await expect(ocrApi.processImage(mockImageBlob)).rejects.toThrow('HTTP error! status: 400')
    })
  })

  describe('processImageBase64', () => {
    it('should send base64 image to API', async () => {
      const mockBase64 = 'data:image/png;base64,iVBORw0KGgoAAAANS...'
      const mockResponseData = {
        image_id: 'test-456',
        text: 'XYZ',
        average_confidence: 0.88,
        processing_time: 0.150,
        timestamp: '2024-01-01T13:00:00Z',
        characters: []
      }
      
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponseData
      } as Response)
      
      const result = await ocrApi.processImageBase64(mockBase64)
      
      expect(global.fetch).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/ocr/process-base64`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ image: mockBase64.split(',')[1] })
        }
      )
      
      expect(result).toEqual(mockResponseData)
    })

    it('should extract base64 data from data URL', async () => {
      const mockBase64WithPrefix = 'data:image/png;base64,ABC123=='
      const mockResponseData = { text: 'test' }
      
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponseData
      } as Response)
      
      await ocrApi.processImageBase64(mockBase64WithPrefix)
      
      const callArgs = vi.mocked(global.fetch).mock.calls[0]
      expect(callArgs[1]?.body).toEqual(JSON.stringify({ image: 'ABC123==' }))
    })
  })

  describe('getHealth', () => {
    it('should check API health status', async () => {
      const mockResponseData = {
        status: 'healthy',
        version: '0.1.0',
        timestamp: '2024-01-01T12:00:00Z'
      }
      
      vi.mocked(global.fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponseData
      } as Response)
      
      const result = await ocrApi.getHealth()
      
      expect(global.fetch).toHaveBeenCalledWith(`${mockApiUrl}/api/v1/health`)
      expect(result).toEqual(mockResponseData)
    })
  })
})