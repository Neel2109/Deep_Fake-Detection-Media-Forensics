import pytest
from PIL import Image

from forensics.exif_analyzer import analyze_exif, gps_decimal_coordinate


def test_analyze_exif_returns_selected_tags_and_explicit_absence():
    image = Image.new("RGB", (8, 6), "navy")
    assert analyze_exif(image) == {
        "has_exif": False,
        "exif_fields": {},
        "exif_tag_count": 0,
        "exif_parse_status": "absent",
    }

    image.getexif()[305] = "Camera Utility"
    result = analyze_exif(image)

    assert result["has_exif"] is True
    assert result["exif_fields"] == {"software": "Camera Utility"}
    assert result["exif_parse_status"] == "available"


def test_gps_decimal_coordinate_checks_axis_and_coordinate_ranges():
    assert gps_decimal_coordinate((40, 30, 0), "N", "latitude") == 40.5
    assert gps_decimal_coordinate((73, 59, 0), "W", "longitude") == -73.9833333
    assert gps_decimal_coordinate((91, 0, 0), "N", "latitude") is None
    assert gps_decimal_coordinate((40, 60, 0), "N", "latitude") is None
    with pytest.raises(ValueError, match="axis"):
        gps_decimal_coordinate((0, 0, 0), "N", "altitude")
