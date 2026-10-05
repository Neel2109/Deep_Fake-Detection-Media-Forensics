"""Smoke tests for the modular backend scaffold and implemented adapters."""

import importlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from forensics.file_validator import validate_file
from forensics.hash_analyzer import sha256_file
from scaffold import FeatureNotConfigured, require_implementation


class BackendScaffoldTests(unittest.TestCase):
    def test_python_modules_are_importable(self):
        project_root = Path(__file__).resolve().parents[1]
        for source in project_root.rglob("*.py"):
            relative_parts = source.relative_to(project_root).parts
            if any(
                part in {"app", "tests", ".venv", ".backend-venv", "venv", "node_modules"}
                for part in relative_parts
            ):
                continue
            module_name = ".".join(source.relative_to(project_root).with_suffix("").parts)
            if module_name == "__init__":
                continue
            if module_name.endswith(".__init__"):
                module_name = module_name.removesuffix(".__init__")
            if module_name:
                with self.subTest(module=module_name):
                    importlib.import_module(module_name)

    def test_file_validator_checks_supported_image_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.png"
            Image.new("RGB", (2, 2), color="white").save(image_path)

            result = validate_file(image_path, image_path.name)

            self.assertEqual(result["media_type"], "image")
            self.assertTrue(result["signature_recognized"])
            self.assertTrue(result["extension_matches_signature"])
            self.assertEqual(len(sha256_file(image_path)), 64)

    def test_file_validator_rejects_extension_signature_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "evidence.jpg"
            Image.new("RGB", (2, 2), color="white").save(image_path, format="PNG")

            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_file(image_path, image_path.name)

    def test_unimplemented_features_fail_explicitly(self):
        with self.assertRaises(FeatureNotConfigured):
            require_implementation("untrained video detector")


if __name__ == "__main__":
    unittest.main()
