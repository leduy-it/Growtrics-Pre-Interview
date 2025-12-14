"""Question detection agent - identifies questions and their structure."""

import json
import re
from typing import List, Optional

from ..models import Question, QuestionPart, AnswerType, ContentFormat
from ..utils.gemini_client import GeminiClient


QUESTION_DETECTION_PROMPT = """Analyze this exam page and extract ALL questions with their complete structure.

## Reasoning Steps (Chain-of-Thought)
1. **Scan**: List all visible question numbers and their locations.
2. **Layout Analysis**: Determine if the page is single or double column.
3. **Structure Analysis**: For each question, identify if it has sub-parts (a, b, i, ii) and how they are nested.
4. **Content Extraction**: Read the text carefully. Identify math formulas for LaTeX conversion.
5. **Confidence Evaluation**: Assess how clear the text and structure are.

## Few-Shot Examples (Use these as guides)

**Example 1: Diagram Embedded in Text**
*Input*: Text says "Refer to Figure 1 below." with a triangle image.
*Output*:
{
    "question_number": "3",
    "question_parts": [{
        "part_id": "a",
        "question_text": "Refer to Figure 1 below. Calculate the area of the triangle.",
        "content_format": "text",
        "diagram_description": "Figure 1: Triangle with base 5cm and height 4cm"
    }]
}

**Example 2: Complex Nesting**
*Input*: "1. (a) Solve: (i) 2x = 4"
*Output*:
{
    "question_number": "1",
    "question_parts": [{
        "part_id": "a",
        "sub_parts": [{
            "part_id": "i",
            "question_text": "Solve: $2x=4$",
            "content_format": "latex"
        }]
    }]
}

## Extraction Rules
For each question you find:
1. Identify the main question number (e.g., 1, 2, 3, 4)
2. Identify deeper hierarchical structure (1 -> a -> i). Use 'sub_parts' for nesting.
3. Extract the COMPLETE question text. Use LaTeX for ANY mathematical expressions (e.g., $x^2 + y^2$).
4. Determine the expected answer type (multiple_choice, short_answer, long_answer, working).
5. Assign a 'confidence' score (0.0 to 1.0) based on extraction certainty.
6. Set 'content_format' to 'latex' if math is present, else 'text'.

Return valid JSON in this structure:
{
    "questions": [
        {
            "question_number": "1",
            "page_numbers": [1],
            "question_parts": [
                {
                    "part_id": "a",
                    "question_text": "Calculate $x$...",
                    "content_format": "latex",
                    "answer_type": "working",
                    "marks": 2,
                    "confidence": 0.95,
                    "sub_parts": [
                        {
                            "part_id": "i",
                            "question_text": "If $y=2$...",
                            "content_format": "latex",
                            "confidence": 0.9
                        }
                    ]
                }
            ]
        }
    ],
    "page_continuation": false
}

Analyze carefully following the reasoning steps. Use LaTeX for math. Nest parts correctly."""


class QuestionDetector:
    """Detects and structures questions from exam page images."""
    
    AGENT_NAME = "question_detector"
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
    
    async def detect_questions(
        self,
        page_images: List[bytes],
        extracted_text: Optional[str] = None
    ) -> List[Question]:
        """
        Detect questions from page images processing in parallel.
        
        Args:
            page_images: List of page images as bytes
            extracted_text: Optional pre-extracted text for context
            
        Returns:
            List of detected Question objects
        """
        all_questions = []
        import asyncio
        
        # Semaphore to limit concurrent requests (prevent instant rate limit)
        sem = asyncio.Semaphore(5)
        
        async def process_page(page_num: int, page_image: bytes):
            async with sem:
                prompt = QUESTION_DETECTION_PROMPT
                
                if extracted_text:
                    prompt += f"\n\nExtracted text for reference:\n{extracted_text}"
                
                response = await self.client.generate_with_image(
                    prompt=prompt,
                    images=[page_image],
                    agent_name=self.AGENT_NAME,
                    system_instruction="You are an expert at analyzing exam papers and extracting structured question data. Always return valid JSON."
                )
                
                # Parse response
                return self._parse_response(response["text"], page_num + 1)

        # Create tasks for all pages
        tasks = [
            process_page(i, img) 
            for i, img in enumerate(page_images)
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        for page_questions in results:
            all_questions.extend(page_questions)
        
        # Merge questions that span multiple pages
        merged_questions = self._merge_questions(all_questions)
        
        return merged_questions
    
    def _parse_response(self, response_text: str, page_num: int) -> List[Question]:
        """Parse Gemini response into Question objects."""
        questions = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            
            for q_data in data.get("questions", []):
                parts = []
                for part_data in q_data.get("question_parts", []):
                    parts.append(self._parse_question_part(part_data))
                
                question = Question(
                    question_number=str(q_data.get("question_number", "")),
                    question_parts=parts,
                    page_numbers=q_data.get("page_numbers", [page_num])
                )
                questions.append(question)
                
        except (json.JSONDecodeError, ValueError) as e:
            # Log error but don't fail - return what we can
            print(f"Warning: Failed to parse question response: {e}")
        
        return questions

    def _parse_question_part(self, data: dict) -> QuestionPart:
        """Recursive parsing of question parts."""
        # Map answer type
        answer_type_str = data.get("answer_type", "short_answer")
        try:
            answer_type = AnswerType(answer_type_str)
        except ValueError:
            answer_type = AnswerType.SHORT_ANSWER

        # Map content format
        content_format_str = data.get("content_format", "text")
        try:
            content_format = ContentFormat(content_format_str)
        except ValueError:
            content_format = ContentFormat.TEXT
        
        # Parse sub-parts recursively
        sub_parts = [
            self._parse_question_part(sub) 
            for sub in data.get("sub_parts", [])
        ]

        return QuestionPart(
            part_id=str(data.get("part_id", "")),
            question_text=data.get("question_text", ""),
            content_format=content_format,
            confidence=float(data.get("confidence", 1.0)),
            answer_type=answer_type,
            marks=data.get("marks"),
            diagrams=[],
            sub_parts=sub_parts
        )
    
    def _merge_questions(self, questions: List[Question]) -> List[Question]:
        """Merge questions that span multiple pages."""
        merged = {}
        
        for q in questions:
            q_num = q.question_number
            
            if q_num in merged:
                # Merge parts and page numbers
                merged[q_num].question_parts.extend(q.question_parts)
                merged[q_num].page_numbers.extend(q.page_numbers)
                merged[q_num].page_numbers = list(set(merged[q_num].page_numbers))
            else:
                merged[q_num] = q
        
        return list(merged.values())
