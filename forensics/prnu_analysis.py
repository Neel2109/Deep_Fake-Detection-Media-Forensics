"""
Photo-Response Non-Uniformity (PRNU) and Sensor Noise Analysis.
Detects local inconsistencies in sensor noise that may indicate splicing or inpainting.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "prnu_analysis"

def estimate_prnu(image_path: str | Path) -> Dict[str, Any]:
    """
    Estimates the sensor noise residual and checks for local inconsistency.
    Without a reference camera fingerprint, this looks for regions where the
    noise pattern deviates significantly from the rest of the image.
    """
    try:
        with Image.open(image_path) as source:
            img = ImageOps.exif_transpose(source).convert("L")
            
            # For performance on huge images, we can downscale slightly,
            # but PRNU is a high-frequency feature, so we try to keep it 
            # as large as feasible. Max 1024 for this analysis.
            max_size = 1024
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            grayscale = np.array(img, dtype=np.float32)
            
        # 1. Extract noise residual
        # A common way to get sensor noise is subtracting a denoised version
        # from the original image.
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        blurred = np.array(blurred_img, dtype=np.float32)
        
        noise_residual = grayscale - blurred
        
        # 2. Local noise variance analysis
        # Authentic images have relatively uniform sensor noise across regions of similar luminance.
        # Deepfakes (especially GANs/Diffusion) often have completely different noise 
        # characteristics in generated regions (e.g. faces).
        
        h, w = noise_residual.shape
        grid_y, grid_x = 4, 4
        step_y, step_x = h // grid_y, w // grid_x
        
        if step_y == 0 or step_x == 0:
            return {"status": "error", "error": "Image too small for PRNU analysis", "inconsistency_score": 0.0}
            
        variances = []
        
        # Calculate variance of noise residual for each block
        for j in range(grid_y):
            for i in range(grid_x):
                block = noise_residual[j*step_y:(j+1)*step_y, i*step_x:(i+1)*step_x]
                if block.size > 0:
                    var = np.var(block)
                    variances.append(var)
                    
        variances = np.array(variances)
        
        # If noise is perfectly uniform, variance of variances is low.
        # If one region is spliced or generated, its noise variance will be an outlier.
        global_var = np.var(noise_residual)
        var_of_vars = np.std(variances)
        
        # Coefficient of variation for noise variances
        cv = var_of_vars / (np.mean(variances) + 1e-6)
        
        # Rough heuristic for inconsistency
        inconsistency_score = min(1.0, float(cv / 1.5))
        
        return {
            "status": "completed",
            "global_noise_variance": float(global_var),
            "local_variance_std": float(var_of_vars),
            "coefficient_of_variation": float(cv),
            "inconsistency_score": float(inconsistency_score),
            "interpretation": "Analyzed local noise residuals. High inconsistency score may indicate that different parts of the image came from different sensors (splicing) or are synthetic."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "inconsistency_score": 0.0
        }
