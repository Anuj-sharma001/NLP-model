"""
Script to generate the comprehensive SAKYTI Multilingual NLP Architecture and Workflow PDF.
Covers:
- System Architecture Overview (Task 1 to Task 9)
- FastAPI Request Lifecycle & Internal Mechanics
- 7-Stage Multilingual NLP Processing Pipeline
- IndicLID & IndicTrans2 ML Models Deep-Dive
- Ayurvedic Terminology Shielding Flow
- API Specifications & Observability
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "SAKYTI Multilingual NLP Service — Architecture & Workflow Specification")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — SAKYTI HEALTHCARE AYURVEDA ECOSYSTEM")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()

def build_pdf(filename="SAKYTI_NLP_Architecture_and_Workflow.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2C5282"),
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C")
    )

    tag_style = ParagraphStyle(
        'BadgeText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2D3748")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1A365D")
    )

    story = []

    # ==================== PAGE 1: TITLE & SYSTEM OVERVIEW ====================
    story.append(Paragraph("SAKYTI Multilingual NLP Architecture", title_style))
    story.append(Paragraph("End-to-End Multilingual Healthcare NLP, Ayurvedic Terminology Shielding & IndicTrans2 Neural Pipeline", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3182CE"), spaceAfter=12))

    story.append(Paragraph("1. Executive Summary & Architectural Overview", h1_style))
    story.append(Paragraph(
        "The <b>SAKYTI Multilingual NLP Service</b> is a production-grade asynchronous microservice developed using <b>FastAPI</b>. "
        "It provides high-precision natural language understanding across <b>22 official Indic languages</b>, detects and transliterates "
        "mixed-script <b>Romanized (Hinglish)</b> input, protects classical <b>Ayurvedic clinical terminology</b> (e.g., <i>Vata</i>, <i>Pitta</i>, <i>Kapha</i>, <i>Agni</i>, <i>Ama</i>) "
        "from literal machine translation corruption, and provides bidirectional neural translation with <b>AI4Bharat IndicTrans2</b> models.",
        body_style
    ))

    # Architecture High-Level Component Flow Table
    arch_flow = [
        [
            Paragraph("<b>CLIENT LAYER</b>", table_header),
            Paragraph("Web UI, Mobile Apps, Clinical Chatbots, Diagnostic Triage Tools sending REST payloads with optional <code>X-Request-ID</code>.", table_cell)
        ],
        [
            Paragraph("<b>GATEWAY & MIDDLEWARE</b>", table_header),
            Paragraph("FastAPI ASGI application with <code>RequestIDMiddleware</code> (contextvars correlation), <code>CORSMiddleware</code>, and strict Pydantic v2 schema validators.", table_cell)
        ],
        [
            Paragraph("<b>DOMAIN SERVICES LAYER</b>", table_header),
            Paragraph("• <b>Normalizer</b>: Unicode NFC, whitespace, punctuation & danda (।)<br/>• <b>IndicLID</b>: FastText dual-model language & script identification<br/>• <b>Romanized Engine</b>: Hinglish lexical detection & term-shielded transliteration<br/>• <b>Terminology Service</b>: 14 core Ayurvedic concepts protection<br/>• <b>Translator</b>: Pure Python IndicProcessor + IndicTrans2 Transformer", table_cell)
        ],
        [
            Paragraph("<b>MODELS & KNOWLEDGE BASES</b>", table_header),
            Paragraph("• <code>IndicLID-FTN / IndicLID-FTR</code> (.bin FastText classifiers)<br/>• <code>IndicTrans2-indic-en-dist-200M / en-indic-dist-200M</code> (RoPE Seq2Seq)<br/>• <code>data/ayurveda_terms.json</code> (Classical taxonomy & provenance)<br/>• <code>data/languages.json</code> (Canonical language registry with Sanskrit)", table_cell)
        ],
        [
            Paragraph("<b>DOWNSTREAM ECOSYSTEM</b>", table_header),
            Paragraph("SAKYTI Diagnostic LLM, Clinical Knowledge Graph, and Ayurvedic Electronic Health Records (EHR).", table_cell)
        ]
    ]

    t_arch = Table(arch_flow, colWidths=[150, 354])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EBF8FF")),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # High-level diagram box
    diag_summary = [
        [Paragraph("<b>HIGH-LEVEL DATA FLOW</b>", ParagraphStyle('HFlow', parent=table_header, textColor=colors.HexColor("#2C5282")))],
        [Paragraph(
            "<b>User Query</b> [e.g. <i>'mujhe pet me dard hai aur pitta vikriti lagti hai'</i>]<br/>"
            "&nbsp;&nbsp;▼<br/>"
            "<b>FastAPI Gateway</b> [Request-ID Injection ➔ Pydantic Validation ➔ Router Dispatch]<br/>"
            "&nbsp;&nbsp;▼<br/>"
            "<b>SAKYTI 7-Stage NLP Pipeline</b> [Normalize ➔ IndicLID ➔ Romanized Handle ➔ Term Extract ➔ IndicTrans2 ➔ IR]<br/>"
            "&nbsp;&nbsp;▼<br/>"
            "<b>Standard Intermediate Representation</b> [Language: Hindi | Script: Latin/Romanized | Terms: Pitta, Vikriti | English: <i>'I have abdominal pain and feel Pitta Vikriti'</i>]",
            body_style
        )]
    ]
    t_diag_sum = Table(diag_summary, colWidths=[504])
    t_diag_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#EDF2F7")),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor("#FFFFFF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_diag_sum)

    # ==================== PAGE 2: FASTAPI REQUEST LIFECYCLE ====================
    story.append(PageBreak())
    story.append(Paragraph("2. FastAPI Request Lifecycle & Internal Mechanics", h1_style))
    story.append(Paragraph(
        "This section details how an HTTP request enters the microservice, gets transformed through ASGI middlewares, "
        "undergoes Pydantic validation, executes through asynchronous lifecycles, and formats structured error responses.",
        body_style
    ))

    fastapi_steps = [
        [
            Paragraph("<b>Step / Component</b>", table_header),
            Paragraph("<b>Internal Mechanics & Behavior</b>", table_header),
            Paragraph("<b>Error Handling / Guardrails</b>", table_header)
        ],
        [
            Paragraph("<b>1. ASGI Server<br/>(Uvicorn)</b>", table_cell),
            Paragraph("Receives raw TCP connection, parses HTTP/1.1 or WebSocket protocol, forwards scope, receive, send ASGI events.", table_cell),
            Paragraph("Rejects malformed HTTP packets; manages async event loops.", table_cell)
        ],
        [
            Paragraph("<b>2. RequestID Middleware</b>", table_cell),
            Paragraph("Inspects <code>X-Request-ID</code> header. If missing, generates a cryptographic UUIDv4. Binds it to Python <code>contextvars</code> and <code>request.state.request_id</code>.", table_cell),
            Paragraph("Guarantees 100% request correlation across all log statements and response headers.", table_cell)
        ],
        [
            Paragraph("<b>3. CORS Middleware</b>", table_cell),
            Paragraph("Evaluates request Origin against configured allowlist. Handles HTTP <code>OPTIONS</code> preflight requests immediately.", table_cell),
            Paragraph("Blocks unauthorized cross-origin requests in production.", table_cell)
        ],
        [
            Paragraph("<b>4. Route Matcher & Dependencies</b>", table_cell),
            Paragraph("Resolves path against APIRouter (e.g. <code>POST /api/v1/nlp/analyze</code>). Evaluates FastAPI dependencies (db sessions, settings).", table_cell),
            Paragraph("Returns standard <code>404 Not Found</code> or <code>405 Method Not Allowed</code> with structured envelope.", table_cell)
        ],
        [
            Paragraph("<b>5. Pydantic v2 Schema Validator</b>", table_cell),
            Paragraph("Deserializes JSON payload into typed model (e.g. <code>NLPAnalyzeRequest</code>). Enforces text length (≤5,000 chars), non-emptiness, thresholds.", table_cell),
            Paragraph("<b>HTTP 422 Unprocessable Entity</b>: Caught by <code>validation_exception_handler</code> returning RFC-7807 <code>ErrorResponse</code>.", table_cell)
        ],
        [
            Paragraph("<b>6. Lifespan Model Singletons</b>", table_cell),
            Paragraph("FastAPI <code>@asynccontextmanager lifespan</code> loads IndicLID FastText and IndicTrans2 PyTorch weights once at boot. Avoids per-request re-initialization.", table_cell),
            Paragraph("Detects CUDA availability; falls back to CPU automatically. Models are pinned to memory.", table_cell)
        ],
        [
            Paragraph("<b>7. Service Execution</b>", table_cell),
            Paragraph("Dispatches request to <code>SAKYTIMultilingualPipeline</code>. Employs async executors or non-blocking threads for model inference.", table_cell),
            Paragraph("Any unhandled exception is caught by global exception handler, logging stacktrace and returning <b>HTTP 500</b>.", table_cell)
        ],
        [
            Paragraph("<b>8. Response Formatting & Headers</b>", table_cell),
            Paragraph("Serializes Pydantic response model to JSON. Middleware intercepts outgoing response, injecting <code>X-Request-ID</code> header.", table_cell),
            Paragraph("Client receives deterministic JSON with latency telemetry (<code>execution_time_ms</code>).", table_cell)
        ]
    ]

    t_fastapi = Table(fastapi_steps, colWidths=[100, 244, 160])
    t_fastapi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_fastapi)
    story.append(Spacer(1, 12))

    story.append(Paragraph("FastAPI Unified Error Envelope Architecture", h2_style))
    story.append(Paragraph(
        "All HTTP exceptions (400, 404, 413, 422, 500, 503) are caught by custom exception handlers registered in <code>app/main.py</code>. "
        "They return a strictly structured JSON envelope matching the <code>ErrorResponse</code> Pydantic schema:",
        body_style
    ))

    error_schema_sample = (
        "{\n"
        '  "error": {\n'
        '    "code": "VALIDATION_ERROR",\n'
        '    "message": "Request payload validation failed.",\n'
        '    "details": [{"loc": ["body", "text"], "msg": "Input should be a valid string", "type": "string_type"}],\n'
        '    "request_id": "c9b1f2a3-8d4e-4b6a-9f1c-7e5d2a8b3c1d",\n'
        '    "timestamp": "2026-09-08T18:30:00Z"\n'
        "  }\n"
        "}"
    )
    t_err = Table([[Paragraph(f"<pre>{error_schema_sample}</pre>", code_style)]], colWidths=[504])
    t_err.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_err)

    # ==================== PAGE 3: 7-STAGE PIPELINE WORKFLOW ====================
    story.append(PageBreak())
    story.append(Paragraph("3. SAKYTI 7-Stage Multilingual NLP Pipeline", h1_style))
    story.append(Paragraph(
        "The core linguistic intelligence engine executes across seven sequential, decoupled stages inside <code>services/pipeline.py</code>. "
        "Each stage transforms the input query while maintaining immutability and provenance tracking.",
        body_style
    ))

    pipeline_stages = [
        [
            Paragraph("<b>Stage</b>", table_header),
            Paragraph("<b>Subsystem & Method</b>", table_header),
            Paragraph("<b>Detailed Functionality & Transformation</b>", table_header)
        ],
        [
            Paragraph("<b>Stage 1:<br/>Input Validation</b>", table_cell),
            Paragraph("<code>Pipeline._validate_input</code><br/>(Task 1 & 8)", table_cell),
            Paragraph("• Verifies non-empty string input and character length limits (&le; 5,000 characters).<br/>• Sanitizes control characters and null bytes.<br/>• Protects downstream PyTorch models from GPU memory exhaustion.", table_cell)
        ],
        [
            Paragraph("<b>Stage 2:<br/>Multilingual Normalization</b>", table_cell),
            Paragraph("<code>services/normalizer.py</code><br/>(Task 6)", table_cell),
            Paragraph("• Applies Unicode NFC canonical decomposition/recomposition.<br/>• Decodes HTML entities and normalizes whitespace.<br/>• Collapses repeated punctuation (e.g. <i>'!!!!'</i> &rarr; <i>'!'</i>).<br/>• <b>Crucial</b>: Preserves sacred Indic punctuation including Devanagari Danda (।), Double Danda (॥), and medical abbreviations.", table_cell)
        ],
        [
            Paragraph("<b>Stage 3:<br/>Language & Script ID</b>", table_cell),
            Paragraph("<code>services/language_detector.py</code><br/>(Task 2 & 3)", table_cell),
            Paragraph("• Inspects Unicode script blocks.<br/>• Routes native Indic text to <b>IndicLID-FTN</b> (22 languages).<br/>• Routes Latin/Romanized text to <b>IndicLID-FTR</b> to detect Romanized Indic vs English.<br/>• Returns canonical language code, script, and confidence.", table_cell)
        ],
        [
            Paragraph("<b>Stage 4:<br/>Romanized Input Handling</b>", table_cell),
            Paragraph("<code>services/romanized_processor.py</code><br/>(Task 7)", table_cell),
            Paragraph("• Evaluates Romanized Indic confidence using lexical markers (<i>'dard'</i>, <i>'pet'</i>, <i>'hai'</i>).<br/>• Transliterates Romanized text to native Devanagari.<br/>• Shields known Ayurvedic terms so phonetic transliteration does not alter medical spellings.<br/>• Produces converted text for subsequent pipeline stages.", table_cell)
        ],
        [
            Paragraph("<b>Stage 5:<br/>Ayurvedic Term Extraction</b>", table_cell),
            Paragraph("<code>services/terminology_service.py</code><br/>(Task 5)", table_cell),
            Paragraph("• Scans text against <code>data/ayurveda_terms.json</code> using boundary-aware regex.<br/>• Extracts character spans, Sanskrit forms, categories (Dosha, Dhatu, Agni, Herbology).<br/>• Tags concepts with <code>do_not_translate: true</code> for translation protection.", table_cell)
        ],
        [
            Paragraph("<b>Stage 6:<br/>IndicTrans2 Translation</b>", table_cell),
            Paragraph("<code>services/translator.py</code><br/>(Task 4)", table_cell),
            Paragraph("• If target language differs from detected language (typically Indic &rarr; English):<br/>&nbsp;&nbsp;1. Masks Ayurvedic terms with unique placeholders (<code>__AYUR_TERM_0__</code>).<br/>&nbsp;&nbsp;2. Attaches IndicTrans2 language prefix tags (e.g. <code>__hin_Deva__</code>).<br/>&nbsp;&nbsp;3. Runs AI4Bharat RoPE Transformer Seq2Seq inference.<br/>&nbsp;&nbsp;4. Detokenizes and restores canonical clinical terms into target syntax.", table_cell)
        ],
        [
            Paragraph("<b>Stage 7:<br/>IR Assembly & Output</b>", table_cell),
            Paragraph("<code>Pipeline.process_query</code><br/>(Task 8 & 9)", table_cell),
            Paragraph("• Synthesizes standardized Intermediate Representation (IR).<br/>• Bundles original text, normalized text, detected language, confidence, extracted terms with offsets, final English text, and execution latency.", table_cell)
        ]
    ]

    t_pipe = Table(pipeline_stages, colWidths=[80, 130, 294])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_pipe)

    # ==================== PAGE 4: ML MODELS & AYURVEDA LAYER ====================
    story.append(PageBreak())
    story.append(Paragraph("4. Machine Learning Models & Ayurvedic Shielding", h1_style))
    story.append(Paragraph(
        "A rigorous, technical breakdown of the machine learning architectures, tokenization mechanics, and Ayurvedic clinical shielding.",
        body_style
    ))

    story.append(Paragraph("4.1 IndicLID Language & Script Identification (AI4Bharat)", h2_style))
    story.append(Paragraph(
        "IndicLID utilizes subword n-gram fastText architectures trained across 22 official Indian languages and English. "
        "To avoid cross-script confusion between native scripts and Romanized text, our service employs a script-gated dual-model router: "
        "Unicode code-point analysis directs native Indic characters (U+0900 to U+0DFF) to <code>IndicLID-FTN</code> and Latin characters (U+0041 to U+007A) "
        "to <code>IndicLID-FTR</code>. This prevents English from being misclassified as an Indic language and vice-versa.",
        body_style
    ))

    story.append(Paragraph("4.2 IndicTrans2 Transformer Translation Engine", h2_style))
    story.append(Paragraph(
        "IndicTrans2 is a 200M/1B parameter sequence-to-sequence Transformer architecture incorporating Rotary Position Embeddings (RoPE). "
        "It supports high-fidelity translation across all 22 scheduled Indian languages plus English and Sanskrit.",
        body_style
    ))

    it_features = [
        [
            Paragraph("<b>Feature / Component</b>", table_header),
            Paragraph("<b>Implementation Details in SAKYTI Service</b>", table_header)
        ],
        [
            Paragraph("<b>Pure Python IndicProcessor</b>", table_cell),
            Paragraph("Original AI4Bharat tooling depended on compiled C++ Cython extensions. We authored a 100% pure Python implementation (<code>app/utils/indic_processor.py</code>) providing exact tokenization and script tagging parity with zero native compiler dependencies.", table_cell)
        ],
        [
            Paragraph("<b>Language Tag Prefixing</b>", table_cell),
            Paragraph("Prepends target and source language tags (e.g. <code>__hin_Deva__</code>, <code>__tam_Taml__</code>, <code>__eng_Latn__</code>) at the token boundary, conditioning the decoder to generate grammatically correct target text.", table_cell)
        ],
        [
            Paragraph("<b>Device Orchestration</b>", table_cell),
            Paragraph("Pytorch models are loaded onto CUDA GPUs if available; otherwise dynamically pin to CPU. Memory is allocated once at startup via the FastAPI Lifespan context manager.", table_cell)
        ]
    ]
    t_it = Table(it_features, colWidths=[140, 364])
    t_it.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_it)
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.3 Ayurvedic Terminology Shielding Mechanism", h2_style))
    story.append(Paragraph(
        "Generic machine translation models corrupt ancient Ayurvedic concepts by translating them literally into secular English terms. "
        "The SAKYTI Terminology Layer eliminates this risk via a 4-step preservation protocol:",
        body_style
    ))

    shield_steps = [
        [Paragraph("<b>Step 1: Term Matching</b>", table_header), Paragraph("Aho-Corasick & boundary-aware regex matches canonical terms, Devanagari, and Romanized spellings from <code>data/ayurveda_terms.json</code>.", table_cell)],
        [Paragraph("<b>Step 2: Placeholder Masking</b>", table_header), Paragraph("Matches are substituted with alphanumeric collision-proof boundary tokens: <i>'वात दोष'</i> &rarr; <code>__AYUR_TERM_0__ dosha</code>.", table_cell)],
        [Paragraph("<b>Step 3: Neural Translation</b>", table_header), Paragraph("IndicTrans2 translates the masked sentence. The transformer treats the placeholder as an opaque proper noun.", table_cell)],
        [Paragraph("<b>Step 4: Canonical Restoration</b>", table_header), Paragraph("Placeholders are mapped back to their standardized Sanskrit / English canonical representations: <code>__AYUR_TERM_0__</code> &rarr; <b>Vata</b>.", table_cell)]
    ]
    t_shield = Table(shield_steps, colWidths=[140, 364])
    t_shield.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#FEFCBF")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#ECC94B")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_shield)
    story.append(Spacer(1, 8))

    # Core 14 terms summary
    terms_list = [
        [Paragraph("<b>Concept</b>", table_header), Paragraph("<b>Sanskrit / Hindi</b>", table_header), Paragraph("<b>Category</b>", table_header), Paragraph("<b>Clinical Definition & Erroneous Literal Translation</b>", table_header)],
        [Paragraph("<b>Vata</b>", table_cell), Paragraph("वात (vāta)", table_cell), Paragraph("Dosha", table_cell), Paragraph("Biological kinetic principle. <i>Never translate as 'Air' or 'Wind'.</i>", table_cell)],
        [Paragraph("<b>Pitta</b>", table_cell), Paragraph("पित्त (pitta)", table_cell), Paragraph("Dosha", table_cell), Paragraph("Biological metabolic & thermal principle. <i>Never translate as 'Bile'.</i>", table_cell)],
        [Paragraph("<b>Kapha</b>", table_cell), Paragraph("कफ (kapha)", table_cell), Paragraph("Dosha", table_cell), Paragraph("Biological structural & cohesive principle. <i>Never translate as 'Phlegm'.</i>", table_cell)],
        [Paragraph("<b>Agni</b>", table_cell), Paragraph("अग्नि (agni)", table_cell), Paragraph("Physiology", table_cell), Paragraph("Digestive and metabolic fire. <i>Never translate as physical 'Fire'.</i>", table_cell)],
        [Paragraph("<b>Ama</b>", table_cell), Paragraph("आम (āma)", table_cell), Paragraph("Pathology", table_cell), Paragraph("Endogenous metabolic toxins / undigested material.", table_cell)],
        [Paragraph("<b>Triphala</b>", table_cell), Paragraph("त्रिफला (triphalā)", table_cell), Paragraph("Herbology", table_cell), Paragraph("Synergistic 3-myrobalan formulation. <i>Never translate as 'Three fruits'.</i>", table_cell)],
        [Paragraph("<b>Panchakarma</b>", table_cell), Paragraph("पञ्चकर्म (pañcakarma)", table_cell), Paragraph("Therapy", table_cell), Paragraph("Five classical bio-cleansing therapies. <i>Never translate as 'Five actions'.</i>", table_cell)]
    ]
    t_terms = Table(terms_list, colWidths=[70, 90, 74, 270])
    t_terms.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_terms)

    # ==================== PAGE 5: API REFERENCE & OBSERVABILITY ====================
    story.append(PageBreak())
    story.append(Paragraph("5. Production API Reference & Observability", h1_style))
    story.append(Paragraph(
        "The microservice exposes a standardized REST interface adhering to OpenAPI 3.1 standards. "
        "Every endpoint integrates automatic Pydantic request validation, response serialization, and distributed tracing headers.",
        body_style
    ))

    api_endpoints = [
        [
            Paragraph("<b>Method & Route</b>", table_header),
            Paragraph("<b>Primary Purpose</b>", table_header),
            Paragraph("<b>Payload & Key Response Fields</b>", table_header)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/api/v1/nlp/analyze</code>", table_cell),
            Paragraph("Master 7-stage NLP Pipeline execution for full medical query analysis.", table_cell),
            Paragraph("<b>In:</b> <code>text</code>, <code>target_language</code>, <code>preserve_terms</code><br/><b>Out:</b> Normalized text, detected language, confidence, script, Romanized status, extracted Ayurvedic concepts, translated English text, latency.", table_cell)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/api/v1/nlp/translate</code>", table_cell),
            Paragraph("Direct neural translation between Indic languages and English with terminology shielding.", table_cell),
            Paragraph("<b>In:</b> <code>text</code>, <code>source_language</code>, <code>target_language</code>, <code>preserve_ayurvedic_terms</code><br/><b>Out:</b> <code>translated_text</code>, model version, device, execution time.", table_cell)
        ],
        [
            Paragraph("<b>POST</b><br/><code>/api/v1/nlp/detect-language</code>", table_cell),
            Paragraph("Identifies language and script across 22 Indic orthographies & Romanized variants.", table_cell),
            Paragraph("<b>In:</b> <code>text</code>, <code>confidence_threshold</code><br/><b>Out:</b> <code>language_code</code>, <code>language_name</code>, <code>script</code>, <code>is_romanized</code>, raw model label.", table_cell)
        ],
        [
            Paragraph("<b>GET</b><br/><code>/health</code>", table_cell),
            Paragraph("Liveness, readiness, and model weight initialization status.", table_cell),
            Paragraph("<b>Out:</b> Microservice version, environment, model status (IndicLID, IndicTrans2), device telemetry (CPU/CUDA).", table_cell)
        ]
    ]
    t_api = Table(api_endpoints, colWidths=[120, 140, 244])
    t_api.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_api)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.1 Telemetry, Logging & Observability Architecture", h2_style))
    story.append(Paragraph(
        "• <b>Correlation Tracking (X-Request-ID)</b>: Every inbound request receives or generates an <code>X-Request-ID</code> header. "
        "This ID is injected into Python's async <code>contextvars</code>, propagating through all logging records and API responses.<br/>"
        "• <b>Structured JSON Logging</b>: Production logs are emitted as single-line JSON objects containing <code>timestamp</code>, <code>level</code>, <code>request_id</code>, <code>message</code>, and <code>latency_ms</code>, enabling seamless ingestion into Datadog, Grafana Loki, or Google Cloud Logging.<br/>"
        "• <b>Test Coverage & Verification</b>: The service includes <b>175 automated unit and integration tests</b> validating Unicode normalization, regex edge cases, Romanized transliteration, IndicTrans2 translation accuracy, terminology preservation, and FastAPI HTTP status codes.",
        body_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.2 Complete Language Matrix (Initial Core Subset)", h2_style))
    lang_table = [
        [Paragraph("<b>Language</b>", table_header), Paragraph("<b>Code</b>", table_header), Paragraph("<b>IndicTrans2 Tag</b>", table_header), Paragraph("<b>Native Script</b>", table_header), Paragraph("<b>Domain Role</b>", table_header)],
        [Paragraph("Hindi", table_cell), Paragraph("<code>hi</code>", table_cell), Paragraph("<code>hin_Deva</code>", table_cell), Paragraph("Devanagari", table_cell), Paragraph("Primary User-Facing & Knowledge", table_cell)],
        [Paragraph("Sanskrit", table_cell), Paragraph("<code>sa</code>", table_cell), Paragraph("<code>san_Deva</code>", table_cell), Paragraph("Devanagari", table_cell), Paragraph("Ayurvedic Source & Knowledge Language", table_cell)],
        [Paragraph("Tamil", table_cell), Paragraph("<code>ta</code>", table_cell), Paragraph("<code>tam_Taml</code>", table_cell), Paragraph("Tamil", table_cell), Paragraph("User-Facing Language (Siddha/Ayurveda)", table_cell)],
        [Paragraph("Telugu", table_cell), Paragraph("<code>te</code>", table_cell), Paragraph("<code>tel_Telu</code>", table_cell), Paragraph("Telugu", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Bengali", table_cell), Paragraph("<code>bn</code>", table_cell), Paragraph("<code>ben_Beng</code>", table_cell), Paragraph("Bengali", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Marathi", table_cell), Paragraph("<code>mr</code>", table_cell), Paragraph("<code>mar_Deva</code>", table_cell), Paragraph("Devanagari", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Gujarati", table_cell), Paragraph("<code>gu</code>", table_cell), Paragraph("<code>guj_Gujr</code>", table_cell), Paragraph("Gujarati", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Kannada", table_cell), Paragraph("<code>kn</code>", table_cell), Paragraph("<code>kan_Knda</code>", table_cell), Paragraph("Kannada", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Malayalam", table_cell), Paragraph("<code>ml</code>", table_cell), Paragraph("<code>mal_Mlym</code>", table_cell), Paragraph("Malayalam", table_cell), Paragraph("User-Facing Language (Kerala Ayurveda)", table_cell)],
        [Paragraph("Punjabi", table_cell), Paragraph("<code>pa</code>", table_cell), Paragraph("<code>pan_Guru</code>", table_cell), Paragraph("Gurmukhi", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("Odia", table_cell), Paragraph("<code>or</code>", table_cell), Paragraph("<code>ory_Orya</code>", table_cell), Paragraph("Odia", table_cell), Paragraph("User-Facing Language", table_cell)],
        [Paragraph("English", table_cell), Paragraph("<code>en</code>", table_cell), Paragraph("<code>eng_Latn</code>", table_cell), Paragraph("Latin", table_cell), Paragraph("Pivot Translation & Clinical Model Target", table_cell)]
    ]
    t_langs = Table(lang_table, colWidths=[80, 50, 94, 90, 190])
    t_langs.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_langs)

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    target_pdf = os.path.join(out_dir, "SAKYTI_NLP_Architecture_and_Workflow.pdf")
    build_pdf(target_pdf)
