# SAKYTI Multilingual NLP Engine

> **Production Multilingual NLP Microservice for Indic Languages, Ayurvedic Clinical Terminology Shielding, and Neural Machine Translation.**

---

## Architecture & Documentation Overview

This repository contains the complete implementation of the **SAKYTI Multilingual NLP Microservice** (`nlp-service/`). The service powers clinical and diagnostic understanding across 22 Indic languages, Romanized Hinglish input, classical Ayurvedic concepts, and neural machine translation via AI4Bharat's IndicTrans2.

### Key Resources
- 📖 **Comprehensive Service Documentation**: [`nlp-service/README.md`](nlp-service/README.md)
- 📄 **Architecture & Workflow Diagram PDF**: [`SAKYTI_NLP_Architecture_and_Workflow.pdf`](SAKYTI_NLP_Architecture_and_Workflow.pdf) (also in [`nlp-service/docs/SAKYTI_NLP_Architecture_and_Workflow.pdf`](nlp-service/docs/SAKYTI_NLP_Architecture_and_Workflow.pdf))
- 🧪 **Automated Test Suite**: 175 tests across all components (`nlp-service/tests/`)

---

## System Architecture Summary

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    SAKYTI SYSTEM ARCHITECTURE                                     |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
   Clients (Web / Mobile / Chatbots / EHR)
      │
      ▼
   FastAPI ASGI Application Gateway (app/main.py)
   ├── RequestIDMiddleware (X-Request-ID distributed tracing via Python contextvars)
   ├── CORSMiddleware (Cross-origin security)
   ├── APIRouter (/api/v1/nlp/*, /api/v1/pipeline/*, /health)
   ├── Pydantic v2 Schema Validation Engine
   └── Global RFC-7807 Structured Exception Handlers (ErrorResponse)
      │
      ▼
   SAKYTI 7-Stage NLP Domain Pipeline (services/pipeline.py)
   ├── Stage 1: Input Validation & GPU Length Bounding
   ├── Stage 2: Multilingual Normalization (Unicode NFC, Danda '।' & Punctuation preservation)
   ├── Stage 3: Language & Script Identification (IndicLID Dual FastText: FTN & FTR)
   ├── Stage 4: Romanized Input Handling (Hinglish Detection & Term-Shielded Transliteration)
   ├── Stage 5: Ayurvedic Terminology Layer (14 Classical Concepts Shielding from data/ayurveda_terms.json)
   ├── Stage 6: Neural Translation (Pure Python IndicProcessor + AI4Bharat IndicTrans2 Transformer)
   └── Stage 7: Standard Intermediate Representation (IR) Output
      │
      ▼
   Downstream Clinical LLM & Diagnostic Reasoning Services
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## Quickstart

```bash
# 1. Navigate to nlp-service
cd nlp-service

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1    # On Windows
source .venv/bin/activate       # On Linux/macOS

# 3. Start development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Run tests
pytest -v
```

For full endpoint specifications, payload schemas, model details, and configuration options, see **[`nlp-service/README.md`](nlp-service/README.md)**.
