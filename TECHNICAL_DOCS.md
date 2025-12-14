# IDP Agent - Technical Implementation & Architecture Report

## 1. Executive Summary
This document outlines the technical architecture, design patterns, and implementation details of the Intelligent Document Processing (IDP) Agent. The system is designed to extract structured question data, answer keys, and diagrams from exam PDF documents using a multi-agent workflow.

## 2. Pipeline Architecture
The core pipeline is built using an event-driven workflow (orchestrated via LlamaIndex) that processes PDFs in disjoint stages.

### 2.1 Workflow Stages
1.  **PDF Processing**: Converts PDF pages to high-quality images.
2.  **Parallel Question Detection (Map-Reduce)**:
    *   **Map**: Each page is processed concurrently by the `QuestionDetector` agent.
    *   **Reduce**: Detected questions are aggregated, and cross-page questions are merged.
    *   **Concurrency Control**: A Semaphore limits active requests (e.g., 5 concurrent) to manage API rate limits.
3.  **Diagram Extraction**: The `DiagramExtractor` scans pages for visual elements and links them to specific questions.
4.  **Answer Key Extraction**: The `AnswerKeyExtractor` uses a **Context-Aware** approach, receiving a summary of detected questions to precisely locate and extract corresponding answers.

## 3. Key Techniques & Approaches

### 3.1 Advanced Prompt Engineering
We employ several cognitive prompting strategies to maximize LLM performance:
*   **Chain-of-Thought (CoT)**: Prompts explicitly instruct the model to "Scan", "Analyze Layout", "Analyze Structure", and "Evaluate Confidence" before generating JSON. This internal monologue reduces hallucination and improves structural parsing.
*   **Few-Shot Learning**: We inject concrete examples of "difficult cases" (e.g., diagrams embedded in text, complex nested lists) into the prompt context to guide the model on edge cases.
*   **Visual Reasoning**: The Vision model is asked to explain *why* it drew a specific bounding box (e.g., "Found caption 'Figure 1'"), ensuring groundedness.

### 3.2 Context-Aware Extraction
Blind extraction is error-prone. We switched to a context-aware model where downstream agents (like the Answer Key Extractor) are fed the *outputs* of upstream agents.
*   *Before*: "Find all answers on this page."
*   *After*: "Find the answer for Question 1a (Solve 2x=4) and Question 1b..."

### 3.3 Enhanced Data Schema
To support production-grade use cases, the data model includes:
*   **Recursive Structures**: `QuestionPart` can contain nested `sub_parts` (1 -> a -> i) to arbitrary depth.
*   **Rich Content**: Content is tagged as `TEXT` or `LATEX` to properly handle mathematical expressions.
*   **Confidence Scores**: Every extracted element includes a confidence score (0-1) to flag items for human review.

## 4. Constraints & Design Decisions

### 4.1 Model Access Limitations
**Constraint**: The current implementation is strictly limited to using the **OpenAI GPT-4.1** API availability.
**Impact**: While we have optimized the pipeline for this model, access to state-of-the-art reasoning models (like GPT-5.2, Claude 4.5 Opus, or Gemini 3 Pro) could significantly enhance performance. A wider selection of models would allow for specialized routing (e.g., using a stronger reasoning model for math parsing and a faster model for layout detection).

### 4.2 Diagram Extraction & Multi-Modal Limitations
**Decision**: We deliberately chose **not** to rely on general-purpose Multi-Modal LLMs (VLMs) for the precise *pixel-coordinate* localization of diagrams.
**Reasoning**: Current general-purpose VLMs often struggle with precise bounding box generation. In the short term, we are following a company-recommended heuristic approach combined with standard vision detection. We avoided deploying specialized layout models (like LayoutLM or fine-tuned YOLOs) to keep the architecture lightweight and dependency-free, though this remains an area for future accuracy boosting.

## 5. Schema Reference

### Question Structure
```json
{
  "question_number": "1",
  "question_parts": [
    {
      "part_id": "a",
      "question_text": "Calculate...",
      "content_format": "latex",
      "confidence": 0.95,
      "sub_parts": [...]
    }
  ]
}
```

### Bounding Box
```json
{
  "bounding_box": {
    "x_min": 0.1, "y_min": 0.2,
    "x_max": 0.5, "y_max": 0.6
  },
  "reasoning": "Captures graph labeled Fig 1"
}
```
