# OCR Question Extraction Pipeline

A multi-agent system for extracting structured questions and answer keys from exam PDFs.

## Overview

This pipeline processes PDF exam papers (Mathematics, Science) and extracts:
- Question structure with subparts (1a, 1b, 2a...)
- Diagrams with bounding box detection
- Answer keys with marking schemes
- Cost tracking and agent logging

## Architecture

```
PDF Input → [PDF Processor] → [Question Detector] → [Diagram Extractor] → [Answer Extractor] → JSON Output
              (PyMuPDF)         (Multi-modal LLM)    (Multi-modal LLM)      (Multi-modal LLM)
```

### Pipeline Steps

| Step | Agent | Function |
|------|-------|----------|
| 1 | PDF Processor | Convert PDF to page images (200 DPI) |
| 2 | Question Detector | Detect questions, parts, answer types |
| 3 | Diagram Extractor | Locate diagrams via bounding boxes, crop and save |
| 4 | Answer Extractor | Extract answer keys, map to questions |
| 5 | Validator | Generate final JSON output |

## Data Model

```
Question
├── question_number: "1", "2"
├── page_numbers: [1, 2]
├── question_parts: [
│   └── QuestionPart
│       ├── part_id: "1a"
│       ├── question_text: "Calculate..."
│       ├── diagrams: ["images/p1_1a_diagram.png"]
│       ├── answer_type: WORKING | SHORT_ANSWER | LONG_ANSWER
│       └── marks: 3
│   ]
└── answer_key: [
    └── AnswerKeyPart
        ├── part_id: "1a"
        ├── answer_content: "42"
        └── explanation: "2 marks for working"
    ]
```

## Framework Choice: LlamaIndex Workflow

### Why LlamaIndex over alternatives?

| Framework | Pros | Cons | Our Choice |
|-----------|------|------|------------|
| **LlamaIndex Workflow** | Event-driven, async, timeout handling, step isolation | Newer, less docs | ✅ Selected |
| LangGraph | Popular, good docs | State management overhead | |
| AutoGen | Multi-agent native | Complex for linear pipelines | |
| CrewAI | Simple syntax | Limited customization | |

**Key reasons for LlamaIndex Workflow:**
1. **Event-driven architecture** - Clean step isolation with explicit data flow
2. **Async by default** - Efficient for I/O bound tasks (API calls)
3. **Built-in timeout** - Critical for production reliability
4. **Context management** - Shared state without global variables

## Key Techniques

### 1. Bounding Box Detection
Vision-capable LLMs detect diagram positions using normalized coordinates (0-1), enabling resolution-independent extraction.

```python
# Normalized bbox → pixel crop
x1, y1, x2, y2 = bbox.x_min * width, bbox.y_min * height, ...
diagram = image.crop((x1, y1, x2, y2))
```

### 2. Multi-page Question Merging
Questions spanning multiple pages are detected and merged by matching question numbers across page boundaries.

### 3. Structured JSON Output
Force LLM to return valid JSON schema, parsed into Pydantic models for validation.

### 4. Incremental Progress Saving
JSON is saved after each step, preventing data loss on interruption:
- `questions_detected` → after step 2
- `diagrams_extracted` → after step 3
- `answers_extracted` → after step 4
- `completed` → final output

### 5. Cost Tracking
Per-model pricing tracks API usage:
```python
cost = (input_tokens / 1M) * input_price + (output_tokens / 1M) * output_price
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` file:
```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1
```

## Usage

```bash
# Process all PDFs in data/
python src/main.py

# Process single PDF
python src/main.py --pdf "path/to/exam.pdf" --name "exam_name"
```

## Output Structure

```
outputs/<exam_name>/
├── questions.json     # Final structured output
├── images/            # Extracted diagrams
│   └── p1_1a_diagram.png
├── logs/              # Agent I/O logs
│   ├── 20241214_103000_question_detector_input.log
│   └── 20241214_103005_question_detector_output.log
└── cost_report.json   # API usage and costs
```

## Log Format

Logs capture agent interactions with image bytes omitted:
```
Agent: question_detector
Direction: input
Timestamp: 2024-12-14T10:30:00

Messages:
[1] user:
[image: 125000 bytes; omitted]

Analyze this exam page and extract questions...
```

## Project Structure

```
src/
├── agents/           # Individual processing agents
│   ├── pdf_processor.py
│   ├── question_detector.py
│   ├── diagram_extractor.py
│   └── answer_key_extractor.py
├── config/           # Settings and environment
├── models/           # Pydantic schemas
├── pipeline/         # LlamaIndex workflow orchestration
├── utils/            # Shared utilities
│   ├── openai_client.py
│   ├── cost_tracker.py
│   └── logger.py
└── main.py           # Entry point
```
