"""Extract selected, human-readable EXIF fields from a decoded image."""

import math
from typing import Any

from PIL import Image

EXIF_FIELD_TAGS = {
    270: "image_description",
    271: "camera_make",
    272: "camera_model",
    274: "orientation",
    305: "software",
    306: "date_time",
    315: "artist",
    36867: "date_time_original",
    36868: "date_time_digitized",
}


def gps_decimal_coordinate(value: Any, reference: Any, axis: str) -> float | None:
    """Convert a validated EXIF degrees/minutes/seconds coordinate to decimal."""
    references = {
        "latitude": {"N": 1, "S": -1},
        "longitude": {"E": 1, "W": -1},
    }
    if axis not in references:
        raise ValueError("axis must be 'latitude' or 'longitude'")
    limit = 90 if axis == "latitude" else 180
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    direction = str(reference).upper()
    if direction not in references[axis]:
        return None
    try:
        degrees, minutes, seconds = (float(part) for part in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if (
        not all(math.isfinite(part) for part in (degrees, minutes, seconds))
        or degrees < 0
        or not 0 <= minutes < 60
        or not 0 <= seconds < 60
        or degrees > limit
        or (degrees == limit and (minutes != 0 or seconds != 0))
    ):
        return None
    decimal = degrees + minutes / 60 + seconds / 3600
    return round(decimal * references[axis][direction], 7)


def analyze_exif(image: Image.Image) -> dict[str, Any]:
    """Return selected EXIF tags and validated GPS coordinates for an open image."""
    try:
        exif = image.getexif()
    except (OSError, TypeError, ValueError) as error:
        return {
            "has_exif": False,
            "exif_fields": {},
            "exif_tag_count": 0,
            "exif_parse_status": "unavailable",
            "exif_parse_error": str(error)[:300],
        }

    result: dict[str, Any] = {
        "has_exif": bool(exif),
        "exif_fields": {},
        "exif_tag_count": len(exif),
        "exif_parse_status": "available" if exif else "absent",
    }
    fields = {
        name: str(exif[tag])[:500]
        for tag, name in EXIF_FIELD_TAGS.items()
        if tag in exif
    }
    if fields:
        result["exif_fields"] = fields

    if exif:
        try:
            gps_ifd = exif.get_ifd(34853)
        except (KeyError, OSError, TypeError, ValueError) as error:
            result["exif_parse_status"] = "partial"
            result["exif_parse_error"] = str(error)[:300]
        else:
            if gps_ifd:
                latitude = gps_decimal_coordinate(
                    gps_ifd.get(2), gps_ifd.get(1), "latitude"
                )
                longitude = gps_decimal_coordinate(
                    gps_ifd.get(4), gps_ifd.get(3), "longitude"
                )
                if latitude is not None and longitude is not None:
                    result["gps_coordinates"] = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "source": "EXIF GPS IFD",
                    }
    return result


__all__ = ["EXIF_FIELD_TAGS", "analyze_exif", "gps_decimal_coordinate"]
