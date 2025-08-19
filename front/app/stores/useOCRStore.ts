import { create } from 'zustand'
import { type OCRResponse } from '~/types/ocr'

interface OCRState {
  isProcessing: boolean
  currentResult: OCRResponse | null
  history: OCRResponse[]
  error: string | null
  
  // Actions
  setProcessing: (processing: boolean) => void
  setResult: (result: OCRResponse) => void
  addToHistory: (result: OCRResponse) => void
  setError: (error: string | null) => void
  clearResult: () => void
  clearHistory: () => void
}

export const useOCRStore = create<OCRState>((set) => ({
  isProcessing: false,
  currentResult: null,
  history: [],
  error: null,
  
  setProcessing: (processing) => set({ isProcessing: processing, error: null }),
  
  setResult: (result) => set((state) => ({
    currentResult: result,
    isProcessing: false,
    error: null,
    history: [result, ...state.history].slice(0, 50) // Keep last 50 results
  })),
  
  addToHistory: (result) => set((state) => ({
    history: [result, ...state.history].slice(0, 50)
  })),
  
  setError: (error) => set({ error, isProcessing: false }),
  
  clearResult: () => set({ currentResult: null }),
  
  clearHistory: () => set({ history: [] })
}))