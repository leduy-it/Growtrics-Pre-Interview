"""Pydantic schemas for question and answer key structures."""

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum


class AnswerType(str, Enum):
    """Types of expected answers for a question part."""
    LONG_ANSWER = "long_answer"
    SHORT_ANSWER = "short_answer"
    WORKING = "working"
    MULTIPLE_CHOICE = "multiple_choice"


class BoundingBox(BaseModel):
    """Bounding box coordinates for diagram extraction."""
    x_min: float = Field(..., description="Left edge (0-1 normalized)")
    y_min: float = Field(..., description="Top edge (0-1 normalized)")
    x_max: float = Field(..., description="Right edge (0-1 normalized)")
    y_max: float = Field(..., description="Bottom edge (0-1 normalized)")
    page_number: int = Field(..., description="Page number (1-indexed)")
    
    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel values."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


class ContentFormat(str, Enum):
    """Format of the textual content."""
    TEXT = "text"
    LATEX = "latex"


class QuestionPart(BaseModel):
    """A single part of a question (e.g., 1a, 1b) with recursive support."""
    part_id: str = Field(..., description="Unique identifier like '1a', '2b'")
    question_text: str = Field(..., description="The full question text")
    content_format: ContentFormat = Field(default=ContentFormat.TEXT, description="Format: text or latex")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence (0-1)")
    
    diagrams: List[str] = Field(default_factory=list, description="Relative paths to diagram images")
    answer_type: AnswerType = Field(default=AnswerType.SHORT_ANSWER)
    marks: Optional[int] = Field(None, description="Points for this part")
    
    sub_parts: List["QuestionPart"] = Field(default_factory=list, description="Nested sub-questions")

# Resolve forward reference for recursion
QuestionPart.model_rebuild()


class AnswerKeyPart(BaseModel):
    """Answer key entry for a question part."""
    part_id: str = Field(..., description="Matches QuestionPart.part_id")
    answer_content: Union[str, List[str]] = Field(..., description="The correct answer(s)")
    explanation: Optional[str] = Field(None, description="Marking scheme or explanation")


class Question(BaseModel):
    """A complete question with all its parts and answers."""
    question_number: str = Field(..., description="Main question number like '1', '2'")
    question_parts: List[QuestionPart] = Field(default_factory=list)
    answer_key: List[AnswerKeyPart] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list, description="Pages where this question appears")


class ExamOutput(BaseModel):
    """Complete output schema for an exam PDF."""
    exam_name: str = Field(..., description="Name of the exam")
    source_file: str = Field(..., description="Original PDF filename")
    total_questions: int = Field(..., description="Total number of questions")
    questions: List[Question] = Field(default_factory=list)
    processing_metadata: dict = Field(default_factory=dict, description="Processing stats")
