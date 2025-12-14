"""OCR Question Extraction Workflow using LlamaIndex."""

from typing import List, Optional, Any
from pathlib import Path

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
    Context,
)

from ..agents import PDFProcessor, QuestionDetector, DiagramExtractor, AnswerKeyExtractor
from ..models import Question, ExamOutput
from ..utils import CostTracker, AgentLogger, OpenAIClient


class PDFProcessedEvent(Event):
    """Event emitted after PDF is processed into images."""
    page_images: List[bytes]
    extracted_text: str
    page_count: int


class QuestionsDetectedEvent(Event):
    """Event emitted after questions are detected."""
    page_images: List[bytes]
    questions: List[Question]


class DiagramsExtractedEvent(Event):
    """Event emitted after diagrams are extracted."""
    page_images: List[bytes]
    questions: List[Question]


class AnswersExtractedEvent(Event):
    """Event emitted after answers are extracted."""
    questions: List[Question]



class OCRWorkflow(Workflow):
    """
    Multi-stage workflow for extracting questions from exam PDFs.
    
    Stages:
    1. PDF Processing - Convert to images, extract text
    2. Question Detection - Identify question structure
    3. Diagram Extraction - Extract diagrams using bounding boxes
    4. Answer Extraction - Extract and map answer keys
    5. Validation - Final validation and output
    """
    
    def __init__(
        self,
        pdf_path: str,
        output_dir: str,
        timeout: int = 600,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__(timeout=timeout, verbose=verbose, **kwargs)
        
        # Store configuration
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        
        # Initialize components
        self.cost_tracker = CostTracker()
        
        # OpenAI client for all tasks (vision + text)
        self.client = OpenAIClient(
            output_dir=output_dir,
            cost_tracker=self.cost_tracker
        )
        
        self.logger = AgentLogger(output_dir)
        
        # Track current state for incremental saving
        self._current_questions = []
        self._current_stage = "initialized"
    
    def _save_progress(self, questions: List[Question], stage: str):
        """Save current progress to JSON file incrementally."""
        import json
        from pathlib import Path
        
        self._current_questions = questions
        self._current_stage = stage
        
        output = ExamOutput(
            exam_name=Path(self.pdf_path).stem,
            source_file=Path(self.pdf_path).name,
            total_questions=len(questions),
            questions=questions,
            processing_metadata={
                "stage": stage,
                "completed": stage == "completed",
                **self.cost_tracker.get_summary()
            }
        )
        
        output_path = Path(self.output_dir) / "questions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output.model_dump(), f, indent=2, ensure_ascii=False)
        
        print(f"   [SAVED] Progress saved ({stage})")
    
    @step
    async def process_pdf(self, ctx: Context, ev: StartEvent) -> PDFProcessedEvent:
        """
        Step 1: Process PDF into page images and extract text.
        """
        print(f"[PDF] Processing: {self.pdf_path}")
        
        # Process PDF
        processor = PDFProcessor(dpi=200)
        page_images, extracted_text, page_count = processor.process(self.pdf_path)
        
        print(f"   [OK] Extracted {page_count} pages")
        
        # Log this step
        self.logger.log_output(
            agent_name="pdf_processor",
            messages=[{
                "role": "system",
                "content": f"Processed PDF: {page_count} pages extracted"
            }],
            metadata={
                "pdf_path": self.pdf_path,
                "page_count": page_count,
                "text_length": len(extracted_text)
            }
        )
        
        return PDFProcessedEvent(
            page_images=page_images,
            extracted_text=extracted_text,
            page_count=page_count,
        )
    
    @step
    async def detect_questions(self, ctx: Context, ev: PDFProcessedEvent) -> QuestionsDetectedEvent:
        """
        Step 2: Detect questions and their structure.
        Uses multi-modal LLM to analyze page images.
        """
        print("[STEP 2] Detecting questions...")
        
        detector = QuestionDetector(self.client)
        questions = await detector.detect_questions(
            page_images=ev.page_images,
            extracted_text=ev.extracted_text
        )
        
        total_parts = sum(len(q.question_parts) for q in questions)
        print(f"   [OK] Found {len(questions)} questions with {total_parts} parts")
        
        self._save_progress(questions, "questions_detected")
        
        return QuestionsDetectedEvent(
            page_images=ev.page_images,
            questions=questions,
        )
    
    @step
    async def extract_diagrams(self, ctx: Context, ev: QuestionsDetectedEvent) -> DiagramsExtractedEvent:
        """
        Step 3: Extract diagrams using bounding box detection.
        """
        print("[STEP 3] Extracting diagrams...")
        
        extractor = DiagramExtractor(
            gemini_client=self.client,
            output_dir=self.output_dir
        )
        questions = await extractor.extract_diagrams(
            page_images=ev.page_images,
            questions=ev.questions
        )
        
        total_diagrams = sum(
            len(part.diagrams)
            for q in questions
            for part in q.question_parts
        )
        print(f"   [OK] Extracted {total_diagrams} diagrams")
        
        self._save_progress(questions, "diagrams_extracted")
        
        return DiagramsExtractedEvent(
            page_images=ev.page_images,
            questions=questions,
        )
    
    @step
    async def extract_answers(self, ctx: Context, ev: DiagramsExtractedEvent) -> AnswersExtractedEvent:
        """
        Step 4: Extract answer keys and map to questions.
        """
        print("[STEP 4] Extracting answer keys...")
        
        extractor = AnswerKeyExtractor(self.client)
        questions = await extractor.extract_answers(
            page_images=ev.page_images,
            questions=ev.questions
        )
        
        total_answers = sum(len(q.answer_key) for q in questions)
        print(f"   [OK] Mapped {total_answers} answers")
        
        self._save_progress(questions, "answers_extracted")
        
        return AnswersExtractedEvent(
            questions=questions,
        )
    
    @step
    async def validate_and_output(self, ctx: Context, ev: AnswersExtractedEvent) -> StopEvent:
        """Step 5: Validate output and generate final JSON."""
        print("[STEP 5] Validating output...")
        
        questions = ev.questions
        
        # Save final output with completed status
        self._save_progress(questions, "completed")
        
        # Print cost summary
        self.cost_tracker.print_summary()
        
        # Save cost report
        self.cost_tracker.save_report(self.output_dir)
        
        print(f"   [OK] Generated output with {len(questions)} questions")
        
        # Build final output for return
        output = ExamOutput(
            exam_name="Extracted Exam",
            source_file="",
            total_questions=len(questions),
            questions=questions,
            processing_metadata=self.cost_tracker.get_summary()
        )
        
        return StopEvent(result=output)


async def run_ocr_pipeline(
    pdf_path: str,
    output_dir: str,
    exam_name: str = "exam"
) -> ExamOutput:
    """
    Run the OCR pipeline on a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for output files
        exam_name: Name of the exam
        
    Returns:
        ExamOutput with extracted questions and answers
    """
    import json
    
    # Setup output directories
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "images").mkdir(exist_ok=True)
    (output_base / "logs").mkdir(exist_ok=True)
    
    # Create workflow with config
    workflow = OCRWorkflow(
        pdf_path=pdf_path,
        output_dir=str(output_base),
        timeout=600,
        verbose=True
    )
    
    # Run workflow - StartEvent with no custom args
    result = await workflow.run()
    
    # Update exam name and source file
    result.exam_name = exam_name
    result.source_file = Path(pdf_path).name
    
    # Save output JSON
    output_json = output_base / "questions.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Processing complete!")
    print(f"   Output: {output_json}")
    
    return result

