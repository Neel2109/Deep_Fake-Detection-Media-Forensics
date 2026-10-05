"""
JPEG Forensics.
Extracts quantization tables and analyzes JPEG-specific artifacts.
"""

from pathlib import Path
from typing import Any, Dict

from PIL import Image

IMPLEMENTATION_STATUS = "implemented"
FEATURE_NAME = "jpeg_forensics"

def extract_quantization_tables(image_path: str | Path) -> Dict[str, Any]:
    """
    Extracts JPEG quantization tables. Different software (Photoshop, 
    Facebook, original cameras) use specific quantization tables.
    Detecting a standard software DQT can indicate editing.
    """
    try:
        with Image.open(image_path) as img:
            if img.format != "JPEG":
                return {
                    "status": "not_applicable", 
                    "error": f"Requires JPEG image, got {img.format}",
                    "jpeg_risk": 0.0
                }
                
            # PIL extracts quantization tables if available
            qtables = getattr(img, "quantization", None)
            
            if not qtables:
                return {
                    "status": "not_found", 
                    "error": "No quantization tables found in image",
                    "jpeg_risk": 0.0
                }
                
            # Usually qtables is a dict mapping table index (0 for luminance, 1 for chrominance) 
            # to a list of 64 integers.
            tables = {}
            for idx, table in qtables.items():
                tables[str(idx)] = list(table)
                
            # In a real forensic system, we'd hash these tables and compare
            # against a database of known signatures (e.g. Photoshop CS6, iPhone 12).
            # Here we just compute some basic stats.
            
            lum_table = qtables.get(0, [])
            if lum_table:
                # Higher values in the table mean higher compression / lower quality
                avg_quantization = sum(lum_table) / 64.0
                
                # Estimated quality (very rough heuristic)
                if avg_quantization < 5:
                    est_quality = ">95 (High Quality)"
                elif avg_quantization < 15:
                    est_quality = "80-95 (Medium Quality)"
                else:
                    est_quality = "<80 (Low Quality)"
            else:
                avg_quantization = 0.0
                est_quality = "Unknown"
                
            return {
                "status": "completed",
                "tables_extracted": len(tables),
                "luminance_avg_quantization": float(avg_quantization),
                "estimated_quality": est_quality,
                "tables": tables,
                "jpeg_risk": 0.0,
                "interpretation": "Extracted JPEG quantization tables. In a full system, these are matched against a database to identify the saving software or camera."
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "jpeg_risk": 0.0
        }
