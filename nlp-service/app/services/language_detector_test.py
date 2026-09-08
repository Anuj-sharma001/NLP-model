"""CLI Test Script for SAKYTI IndicLID Language Detection.

Usage:
    python -m app.services.language_detector_test "मुझे पेट में दर्द है"
"""

import json
import sys

# Configure UTF-8 output encoding for cross-platform and Windows terminal support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import get_settings
from app.models.indiclid_model import IndicLIDModelManager
from app.services.language_detector import detect_language


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.services.language_detector_test \"<text_to_classify>\" [confidence_threshold]")
        print("Example: python -m app.services.language_detector_test \"मुझे पेट में दर्द है\"")
        sys.exit(1)

    input_text = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    # Load models once
    settings = get_settings()
    manager = IndicLIDModelManager.get_instance()
    manager.load_models(settings=settings)

    # Detect language
    result = detect_language(input_text, confidence_threshold=threshold, settings=settings)

    output = {
        "input_text": input_text,
        "language_code": result.language_code,
        "language_name": result.language_name,
        "script": result.script,
        "confidence": result.confidence,
        "is_romanized": result.is_romanized,
        "raw_model_result": result.raw_model_result,
    }

    print("\n--- SAKYTI IndicLID Language Detection Result ---")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
