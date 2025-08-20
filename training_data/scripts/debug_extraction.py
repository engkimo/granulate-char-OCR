#!/usr/bin/env python3
"""
Debug script to visualize the granulate character extraction process.
This helps identify why we're getting the yellow "A" instead of the white granulate character.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze_bubble(image_path):
    """Load bubble image and analyze its color channels."""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # BGRA
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    return img, img_rgb

def visualize_color_extraction(img, img_rgb):
    """Visualize different color extraction methods."""
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Granulate Character Extraction Debug', fontsize=16)
    
    # Row 1: Original and color channels
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Split channels
    if len(img.shape) == 3:
        b, g, r = cv2.split(img[:, :, :3])
        axes[0, 1].imshow(r, cmap='Reds')
        axes[0, 1].set_title('Red Channel')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(g, cmap='Greens')
        axes[0, 2].set_title('Green Channel')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(b, cmap='Blues')
        axes[0, 3].set_title('Blue Channel')
        axes[0, 3].axis('off')
    
    # Row 2: HSV analysis
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    axes[1, 0].imshow(h, cmap='hsv')
    axes[1, 0].set_title('Hue Channel')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(s, cmap='gray')
    axes[1, 1].set_title('Saturation Channel')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(v, cmap='gray')
    axes[1, 2].set_title('Value Channel')
    axes[1, 2].axis('off')
    
    # Yellow color detection (for the yellow A)
    # Yellow in HSV: Hue around 20-30, high saturation and value
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    axes[1, 3].imshow(yellow_mask, cmap='gray')
    axes[1, 3].set_title('Yellow Mask (Yellow A)')
    axes[1, 3].axis('off')
    
    # Row 3: White/granulate detection methods
    # Method 1: High value, low saturation (typical for white/gray)
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    
    axes[2, 0].imshow(white_mask, cmap='gray')
    axes[2, 0].set_title('White/Gray Mask (Granulate)')
    axes[2, 0].axis('off')
    
    # Method 2: Grayscale thresholding
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    axes[2, 1].imshow(thresh1, cmap='gray')
    axes[2, 1].set_title('Grayscale Threshold (>200)')
    axes[2, 1].axis('off')
    
    # Method 3: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    
    axes[2, 2].imshow(adaptive_thresh, cmap='gray')
    axes[2, 2].set_title('Adaptive Threshold')
    axes[2, 2].axis('off')
    
    # Method 4: Exclude yellow and purple, keep the rest
    # Purple detection
    purple_lower1 = np.array([130, 50, 50])
    purple_upper1 = np.array([150, 255, 255])
    purple_lower2 = np.array([160, 50, 50])
    purple_upper2 = np.array([180, 255, 255])
    purple_mask1 = cv2.inRange(hsv, purple_lower1, purple_upper1)
    purple_mask2 = cv2.inRange(hsv, purple_lower2, purple_upper2)
    purple_mask = cv2.bitwise_or(purple_mask1, purple_mask2)
    
    # Combine exclusions
    exclude_mask = cv2.bitwise_or(yellow_mask, purple_mask)
    
    # Create granulate mask by excluding yellow and purple
    granulate_mask = cv2.bitwise_not(exclude_mask)
    
    # Also exclude very dark areas
    _, dark_mask = cv2.threshold(v, 50, 255, cv2.THRESH_BINARY_INV)
    granulate_mask = cv2.bitwise_and(granulate_mask, cv2.bitwise_not(dark_mask))
    
    axes[2, 3].imshow(granulate_mask, cmap='gray')
    axes[2, 3].set_title('Granulate Mask (Exclude Colors)')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    return white_mask, granulate_mask, yellow_mask

def extract_contours_and_visualize(img, mask, title="Contour Detection"):
    """Extract contours from mask and visualize them."""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    
    # Original mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Input Mask')
    axes[0].axis('off')
    
    # Draw all contours
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    axes[1].imshow(contour_img, cmap='gray')
    axes[1].set_title(f'All Contours ({len(contours)} found)')
    axes[1].axis('off')
    
    # Draw largest contours
    if contours:
        largest_contour_img = np.zeros_like(mask)
        # Draw up to 3 largest contours
        for i, contour in enumerate(contours[:3]):
            color = 255 - (i * 80)  # Different gray levels
            cv2.drawContours(largest_contour_img, [contour], -1, color, -1)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(largest_contour_img, (x, y), (x+w, y+h), 255, 2)
            
            # Print contour info
            area = cv2.contourArea(contour)
            print(f"Contour {i+1}: Area={area:.0f}, Bounds=({x},{y},{w},{h})")
        
        axes[2].imshow(largest_contour_img, cmap='gray')
        axes[2].set_title('Largest Contours with Bounding Boxes')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No contours found', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].axis('off')
    
    plt.tight_layout()
    
    return contours

def main():
    # Path to the bubble image
    bubble_path = Path("/Users/ohoriryosuke/2025820_granulate_ocr/granulate-char-OCR/training_data/extracted/A/A_bubble.png")
    
    print(f"Loading bubble image from: {bubble_path}")
    
    # Load and analyze the image
    img, img_rgb = load_and_analyze_bubble(bubble_path)
    
    if img is None:
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    # Visualize color extraction methods
    white_mask, granulate_mask, yellow_mask = visualize_color_extraction(img, img_rgb)
    
    # Extract and visualize contours for different masks
    print("\n--- Yellow Character Contours ---")
    yellow_contours = extract_contours_and_visualize(img, yellow_mask, "Yellow Character Contours")
    
    print("\n--- White/Granulate Character Contours ---")
    granulate_contours = extract_contours_and_visualize(img, granulate_mask, "Granulate Character Contours")
    
    # Final comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Final Extraction Comparison', fontsize=14)
    
    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Extract yellow region
    yellow_extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=yellow_mask)
    axes[1].imshow(yellow_extracted)
    axes[1].set_title('Yellow Character Extracted')
    axes[1].axis('off')
    
    # Extract granulate region
    if len(img_rgb.shape) == 3:
        granulate_extracted = cv2.bitwise_and(img_rgb[:,:,:3], img_rgb[:,:,:3], mask=granulate_mask)
    else:
        granulate_extracted = cv2.bitwise_and(img_rgb, img_rgb, mask=granulate_mask)
    axes[2].imshow(granulate_extracted)
    axes[2].set_title('Granulate Character Extracted')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Show all plots
    plt.show()
    
    print("\nDebug visualization complete!")
    print("The script shows various methods to separate the yellow 'A' from the white granulate character.")
    print("The key is to use color-based segmentation in HSV space to exclude the yellow and purple regions.")

if __name__ == "__main__":
    main()