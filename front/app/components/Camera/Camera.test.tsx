import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Camera } from './Camera'

describe('Camera Component', () => {
  const mockOnCapture = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    // Mock successful camera access
    const mockStream = {
      getTracks: () => [{
        stop: vi.fn()
      }]
    }
    vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(mockStream as any)
  })

  it('should render camera controls', () => {
    render(<Camera onCapture={mockOnCapture} />)
    
    expect(screen.getByRole('button', { name: /capture/i })).toBeInTheDocument()
    expect(screen.getByTestId('camera-video')).toBeInTheDocument()
  })

  it('should request camera permission on mount', async () => {
    render(<Camera onCapture={mockOnCapture} />)
    
    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
        video: { facingMode: 'environment' },
        audio: false
      })
    })
  })

  it('should show error when camera access is denied', async () => {
    vi.mocked(navigator.mediaDevices.getUserMedia).mockRejectedValue(
      new Error('Permission denied')
    )
    
    render(<Camera onCapture={mockOnCapture} />)
    
    await waitFor(() => {
      expect(screen.getByText(/camera access denied/i)).toBeInTheDocument()
    })
  })

  it('should capture image when capture button is clicked', async () => {
    const user = userEvent.setup()
    const mockVideo = {
      videoWidth: 640,
      videoHeight: 480
    }
    
    render(<Camera onCapture={mockOnCapture} />)
    
    // Wait for camera to be ready
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /capture/i })).not.toBeDisabled()
    })
    
    // Mock video dimensions
    const video = screen.getByTestId('camera-video')
    Object.defineProperty(video, 'videoWidth', { value: mockVideo.videoWidth })
    Object.defineProperty(video, 'videoHeight', { value: mockVideo.videoHeight })
    
    // Click capture button
    await user.click(screen.getByRole('button', { name: /capture/i }))
    
    expect(mockOnCapture).toHaveBeenCalledWith(expect.any(Blob))
  })

  it('should switch between front and back camera', async () => {
    const user = userEvent.setup()
    render(<Camera onCapture={mockOnCapture} />)
    
    const switchButton = screen.getByRole('button', { name: /switch camera/i })
    await user.click(switchButton)
    
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenLastCalledWith({
      video: { facingMode: 'user' },
      audio: false
    })
  })

  it('should show loading state while accessing camera', () => {
    render(<Camera onCapture={mockOnCapture} />)
    
    expect(screen.getByText(/accessing camera/i)).toBeInTheDocument()
  })

  it('should clean up camera stream on unmount', async () => {
    const mockStop = vi.fn()
    const mockStream = {
      getTracks: () => [{ stop: mockStop }]
    }
    vi.mocked(navigator.mediaDevices.getUserMedia).mockResolvedValue(mockStream as any)
    
    const { unmount } = render(<Camera onCapture={mockOnCapture} />)
    
    await waitFor(() => {
      expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalled()
    })
    
    unmount()
    
    expect(mockStop).toHaveBeenCalled()
  })
})