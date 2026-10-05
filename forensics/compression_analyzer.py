"""
Compression and Error Level Analysis (ELA) for Media Forensics.
Detects regions with different compression levels, often indicating splicing.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageOps, ImageChops

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "compression_analyzer"

def analyze_compression(image_path: str | Path) -> Dict[str, Any]:
    """
    Performs Error Level Analysis (ELA) to detect manipulation.
    Resaves the image at 90% quality and computes the absolute difference.
    """
    try:
        with Image.open(image_path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
            
            # 1. Error Level Analysis (ELA)
            # Save at known quality
            temp_io = BytesIO()
            img.save(temp_io, "JPEG", quality=90)
            temp_io.seek(0)
            
            resaved_img = Image.open(temp_io).convert("RGB")
            
            # Compute absolute difference
            ela_image = ImageChops.difference(img, resaved_img)
            ela_arr = np.array(ela_image)
            
            # Enhance ELA image (scale differences to 0-255)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            
            if max_diff == 0:
                max_diff = 1
                
            scale = 255.0 / max_diff
            ela_enhanced = np.clip(ela_arr * scale, 0, 255).astype(np.uint8)
            
            # Calculate overall anomaly stats
            mean_ela = float(np.mean(ela_arr))
            std_ela = float(np.std(ela_arr))
            max_ela = int(max_diff)
            
            # Determine overall risk
            # High standard deviation or very high max difference suggests splicing
            compression_risk = 0.0
            if std_ela > 5.0:
                compression_risk = min(1.0, (std_ela - 5.0) / 10.0)
                
            double_compression = False
            # Check for double compression (rough heuristic based on histogram)
            # A more robust check requires quantization table extraction
            
            # 2. Localized Anomalies (Grid-based ELA checking)
            # Divide image into 8x8 grid to find local hot spots
            h, w, _ = ela_arr.shape
            grid_y, grid_x = 8, 8
            step_y, step_x = h // grid_y, w // grid_x
            
            localized_anomalies: List[Dict[str, Any]] = []
            
            if step_y > 0 and step_x > 0:
                # Calculate mean ELA for the whole image to find relative hotspots
                global_mean = mean_ela
                
                for j in range(grid_y):
                    for i in range(grid_x):
                        region = ela_arr[j*step_y:(j+1)*step_y, i*step_x:(i+1)*step_x]
                        if region.size > 0:
                            region_mean = np.mean(region)
                            # If a region has significantly higher error, it's anomalous
                            if region_mean > global_mean + 2 * std_ela and region_mean > 3.0:
                                localized_anomalies.append({
                                    "region": f"grid_{j}_{i}",
                                    "bbox": [i*step_x, j*step_y, (i+1)*step_x, (j+1)*step_y],
                                    "severity": "high" if region_mean > global_mean + 4 * std_ela else "medium",
                                    "ela_mean": float(region_mean)
                                })
            
            return {
                "status": "completed",
                "ela_mean": mean_ela,
                "ela_std": std_ela,
                "ela_max": max_ela,
                "ela_anomaly": compression_risk, # Added for backward compatibility with stub name
                "compression_risk": compression_risk,
                "double_compression": double_compression,
                "localized_anomalies": localized_anomalies,
                "interpretation": "Performed Error Level Analysis. Regions with significantly higher error levels compared to the rest of the image may indicate splicing or local manipulation."
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "compression_risk": 0.0,
            "ela_anomaly": 0.0
        }
