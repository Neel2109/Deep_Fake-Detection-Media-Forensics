"""
Discrete Cosine Transform (DCT) Analysis for Media Forensics.
Extracts block-wise DCT coefficients to identify unnatural frequency distributions.
"""

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageOps

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "dct"

def _dct2(block: np.ndarray) -> np.ndarray:
    """Compute the 2D Discrete Cosine Transform of a block."""
    import scipy.fftpack
    return scipy.fftpack.dct(
        scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho'
    )

def analyze_dct_coefficients(image_path: str | Path) -> Dict[str, Any]:
    """
    Analyzes the DCT coefficients of an image to detect statistical anomalies
    often introduced by AI generation or manipulation.
    """
    try:
        # We need scipy.fftpack for DCT. If not available, fallback gracefully.
        import scipy.fftpack
    except ImportError:
        return {
            "status": "error",
            "error": "scipy is required for DCT analysis (pip install scipy)",
            "anomaly_score": 0.0
        }

    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("L")
            width, height = img.size
            
            # Ensure dimensions are multiples of 8 for block analysis
            w_crop = (width // 8) * 8
            h_crop = (height // 8) * 8
            if w_crop == 0 or h_crop == 0:
                return {"status": "error", "error": "Image too small for block analysis", "anomaly_score": 0.0}
                
            img = img.crop((0, 0, w_crop, h_crop))
            img_arr = np.array(img, dtype=np.float32) - 128.0

        # Process 8x8 blocks
        h_blocks = h_crop // 8
        w_blocks = w_crop // 8
        
        # This can be slow in pure python, so we sample if the image is large
        max_blocks = 1000
        total_blocks = h_blocks * w_blocks
        
        if total_blocks > max_blocks:
            # Randomly sample blocks
            np.random.seed(42) # For reproducibility
            y_idxs = np.random.randint(0, h_blocks, max_blocks)
            x_idxs = np.random.randint(0, w_blocks, max_blocks)
        else:
            y_idxs = np.repeat(np.arange(h_blocks), w_blocks)
            x_idxs = np.tile(np.arange(w_blocks), h_blocks)
            
        ac_coefficients = []
        for y, x in zip(y_idxs, x_idxs):
            block = img_arr[y*8:(y+1)*8, x*8:(x+1)*8]
            dct_block = _dct2(block)
            # Flatten and remove DC component (index 0)
            ac_coeffs = dct_block.flatten()[1:]
            ac_coefficients.extend(ac_coeffs)

        ac_coefficients = np.array(ac_coefficients)
        
        # Analyze AC coefficient distribution
        # Real images typically follow a Laplacian distribution centered at 0
        # AI generated images sometimes have unnatural gaps or spikes
        
        # Very simple anomaly score: check for excessive high-frequency AC components
        # Or check deviation from expected Laplace distribution
        mean_abs_ac = np.mean(np.abs(ac_coefficients))
        
        # Calculate a simple "first digit" (Benford's law-ish) metric for AC coeffs
        # Real images tend to follow Benford's law better than deepfakes
        first_digits = []
        for val in ac_coefficients:
            if abs(val) >= 1:
                first_digits.append(int(str(abs(val))[0]))
        
        anomaly_score = 0.0
        benford_dist = None
        
        if first_digits:
            digit_counts = np.bincount(first_digits, minlength=10)[1:]
            total_digits = len(first_digits)
            actual_dist = digit_counts / total_digits
            
            # Expected Benford distribution
            expected_dist = np.log10(1 + 1 / np.arange(1, 10))
            
            # Chi-square like distance
            dist = np.sum((actual_dist - expected_dist)**2 / expected_dist)
            
            # Normalize to 0-1 range roughly. Higher dist = more anomalous
            anomaly_score = min(1.0, float(dist) * 2.0)
            benford_dist = [float(x) for x in actual_dist]

        return {
            "status": "completed",
            "blocks_analyzed": len(y_idxs),
            "mean_abs_ac_coeff": float(mean_abs_ac),
            "anomaly_score": float(anomaly_score),
            "benford_distribution": benford_dist,
            "interpretation": "Analyzed 8x8 DCT AC coefficients. Higher anomaly score indicates statistical divergence from natural image properties."
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "anomaly_score": 0.0
        }
