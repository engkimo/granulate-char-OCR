import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { OCRResult } from './OCRResult'

describe('OCRResult Component', () => {
  const mockResult = {
    image_id: 'test-123',
    text: 'ABC123',
    average_confidence: 0.92,
    processing_time: 0.256,
    timestamp: '2024-01-01T12:00:00Z',
    characters: [
      { granulate_symbol: 'ᐁ', latin_equivalent: 'A', confidence: 0.95 },
      { granulate_symbol: 'ᐂ', latin_equivalent: 'B', confidence: 0.92 },
      { granulate_symbol: 'ᐃ', latin_equivalent: 'C', confidence: 0.88 },
      { granulate_symbol: 'ᑐ', latin_equivalent: '1', confidence: 0.93 },
      { granulate_symbol: 'ᑑ', latin_equivalent: '2', confidence: 0.91 },
      { granulate_symbol: 'ᑒ', latin_equivalent: '3', confidence: 0.90 }
    ]
  }

  it('should render OCR result text', () => {
    render(<OCRResult result={mockResult} />)
    
    expect(screen.getByText('ABC123')).toBeInTheDocument()
  })

  it('should display average confidence', () => {
    render(<OCRResult result={mockResult} />)
    
    // Find the average confidence specifically in the metrics section
    const metricsSection = screen.getByText('Average Confidence').parentElement
    expect(metricsSection).toHaveTextContent('92%')
  })

  it('should show processing time', () => {
    render(<OCRResult result={mockResult} />)
    
    expect(screen.getByText(/0.256s/i)).toBeInTheDocument()
  })

  it('should render all characters with their mappings', () => {
    render(<OCRResult result={mockResult} />)
    
    // Check granulate symbols
    expect(screen.getByText('ᐁ')).toBeInTheDocument()
    expect(screen.getByText('ᐂ')).toBeInTheDocument()
    expect(screen.getByText('ᐃ')).toBeInTheDocument()
    
    // Check latin equivalents
    mockResult.characters.forEach(char => {
      expect(screen.getByText(char.latin_equivalent)).toBeInTheDocument()
    })
  })

  it('should highlight low confidence characters', () => {
    const resultWithLowConfidence = {
      ...mockResult,
      characters: [
        { granulate_symbol: 'ᐁ', latin_equivalent: 'A', confidence: 0.45 }
      ]
    }
    
    render(<OCRResult result={resultWithLowConfidence} />)
    
    const lowConfidenceElement = screen.getByTestId('character-0')
    expect(lowConfidenceElement).toHaveClass('low-confidence')
  })

  it('should show empty state when no characters detected', () => {
    const emptyResult = {
      ...mockResult,
      text: '',
      characters: []
    }
    
    render(<OCRResult result={emptyResult} />)
    
    expect(screen.getByText(/no characters detected/i)).toBeInTheDocument()
  })

  it('should display confidence for each character', () => {
    render(<OCRResult result={mockResult} />)
    
    // Check that all unique confidence values are displayed
    const confidenceValues = ['95%', '88%', '93%', '91%', '90%']
    confidenceValues.forEach(value => {
      expect(screen.getByText(value)).toBeInTheDocument()
    })
    
    // 92% appears twice (average and character), so we check it exists
    expect(screen.getAllByText('92%')).toHaveLength(2)
  })

  it('should handle undefined result gracefully', () => {
    render(<OCRResult result={undefined} />)
    
    expect(screen.getByText(/no result available/i)).toBeInTheDocument()
  })
})