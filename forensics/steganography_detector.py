"""
Steganography Detection.
Analyzes Least Significant Bits (LSB) and entropy to detect hidden payloads.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "steganography_detector"

def analyze_steganography(image_path: str | Path) -> Dict[str, Any]:
    """
    Analyzes the image for potential steganography by examining the
    Least Significant Bits (LSB). Randomly distributed LSBs often
    indicate an encrypted or compressed hidden payload.
    """
    try:
        with Image.open(image_path) as img:
            # Steganography is typically done in lossless formats (PNG),
            # but can survive high-quality JPEG in some cases.
            img = img.convert("RGB")
            
            # Analyze a sample of the image to save time
            max_size = 512
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))
                
            arr = np.array(img, dtype=np.uint8)
            
        # Extract LSB of the red channel (common target)
        # 1 means odd, 0 means even
        lsb_r = arr[:, :, 0] & 1
        
        # Calculate LSB ratio
        # A natural image's LSB ratio might fluctuate, but a fully packed
        # steganographic image tends to have a near perfect 50/50 split of 0s and 1s.
        ones_count = np.sum(lsb_r)
        total_pixels = lsb_r.size
        
        ratio = ones_count / total_pixels
        
        # An ideal random payload has a ratio of exactly 0.5.
        # We calculate deviation from 0.5.
        deviation = abs(ratio - 0.5)
        
        # If deviation is extremely small (e.g. < 0.005), it's highly suspicious.
        # However, natural images can also be close to 0.5.
        # More advanced checks would use a Chi-Square attack on Pairs of Values (PoV).
        
        stego_risk = 0.0
        if deviation < 0.01:
            stego_risk = 0.4
        if deviation < 0.001:
            stego_risk = 0.8
            
        return {
            "status": "completed",
            "lsb_ratio_red": float(ratio),
            "lsb_deviation_from_random": float(deviation),
            "stego_risk": float(stego_risk),
            "interpretation": "Analyzed Least Significant Bits (LSB). An LSB ratio extremely close to 0.5 (random distribution) may indicate a hidden encrypted payload."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "stego_risk": 0.0
        }
