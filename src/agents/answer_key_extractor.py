"""Answer key extraction agent - extracts and maps answers to questions."""

import json
import re
from typing import List, Dict, Any, Optional

from ..models import Question, AnswerKeyPart, QuestionPart
from ..utils.gemini_client import GeminiClient


ANSWER_KEY_PROMPT = """Analyze this exam page and extract the ANSWER KEY / MARKING SCHEME.

For each answer found:
1. Match it to the question number and part (e.g., 1a, 2b)
2. Extract the correct answer(s)
3. Include any marking scheme, acceptable variations, or explanations

Return your response as valid JSON:
{
    "answers": [
        {
            "part_id": "1a",
            "answer_content": "42",
            "acceptable_variations": ["42.0", "forty-two"],
            "explanation": "2 marks for correct working, 1 mark for final answer"
        },
        {
            "part_id": "2",
            "answer_content": "B",
            "explanation": "Multiple choice - option B is correct"
        }
    ],
    "is_answer_key_page": true,
    "notes": "Any additional observations"
}

If this page does not contain an answer key, return:
{
    "answers": [],
    "is_answer_key_page": false,
    "notes": "No answer key found on this page"
}"""


class AnswerKeyExtractor:
    """Extracts answer keys and maps them to questions."""
    
    AGENT_NAME = "answer_key_extractor"
    
    def __init__(self, gemini_client: GeminiClient):
        self.client = gemini_client
    
    async def extract_answers(
        self,
        page_images: List[bytes],
        questions: List[Question]
    ) -> List[Question]:
        """
        Extract answer keys and attach to questions with context-aware prompting.
        
        Args:
            page_images: List of page images as bytes
            questions: List of detected questions
            
        Returns:
            Updated questions with answer keys filled in
        """
        # Collect all answers from all pages
        all_answers = []
        
        # Build context string from detected questions
        question_context = "Questions to find answers for:\n"
        for q in questions:
            for part in q.question_parts:
                question_context += f"- Q{q.question_number}{part.part_id}: {part.question_text[:50]}...\n"
                # Add sub-parts context
                for sub in part.sub_parts:
                    question_context += f"  - Sub-part ({sub.part_id}): {sub.question_text[:30]}...\n"
        
        for page_num, page_image in enumerate(page_images):
            # Inject context into prompt
            context_prompt = f"{ANSWER_KEY_PROMPT}\n\nCONTEXT:\n{question_context}"
            
            response = await self.client.generate_with_image(
                prompt=context_prompt,
                images=[page_image],
                agent_name=self.AGENT_NAME,
                system_instruction="You are an expert at extracting answer keys and marking schemes from exam papers. Return valid JSON."
            )
            
            answers = self._parse_answer_response(response["text"])
            all_answers.extend(answers)
        
        # Map answers to questions
        questions = self._map_answers_to_questions(questions, all_answers)
        
        return questions
    
    def _parse_answer_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse Gemini response into answer data."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            
            if not data.get("is_answer_key_page", True):
                return []
            
            return data.get("answers", [])
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse answer response: {e}")
            return []
    
    def _map_answers_to_questions(
        self,
        questions: List[Question],
        answers: List[Dict[str, Any]]
    ) -> List[Question]:
        """Map extracted answers to question parts recursively."""
        # Build answer lookup
        answer_lookup = {}
        for ans in answers:
            part_id = ans.get("part_id", "")
            answer_lookup[part_id] = ans
            
            # Also try normalized versions
            normalized = self._normalize_part_id(part_id)
            if normalized != part_id:
                answer_lookup[normalized] = ans
        
        # Attach answers to questions
        for question in questions:
            question.answer_key = []
            
            for part in question.question_parts:
                # Recursive mapping
                self._map_answers_to_part_recursive(part, answer_lookup, question.question_number, question.answer_key)
        
        return questions

    def _map_answers_to_part_recursive(
        self,
        part: QuestionPart,
        answer_lookup: Dict[str, Any],
        question_number: str,
        answer_key_list: List[AnswerKeyPart]
    ):
        """Recursively map answers to parts and sub-parts."""
        # 1. Try to find matching answer for THIS part
        ans_data = answer_lookup.get(part.part_id)
        
        if not ans_data:
            # Try with question number prefix (e.g. "1a")
            ans_data = answer_lookup.get(f"{question_number}{part.part_id}")
        
        if not ans_data:
            # Try normalized version
            normalized = self._normalize_part_id(part.part_id)
            ans_data = answer_lookup.get(normalized)
        
        if ans_data:
            # Build answer content
            content = ans_data.get("answer_content", "")
            variations = ans_data.get("acceptable_variations", [])
            
            if variations:
                content = [content] + variations
            
            answer_part = AnswerKeyPart(
                part_id=part.part_id,
                answer_content=content,
                explanation=ans_data.get("explanation")
            )
            answer_key_list.append(answer_part)

        # 2. Recurse for sub-parts
        for sub_part in part.sub_parts:
            self._map_answers_to_part_recursive(sub_part, answer_lookup, question_number, answer_key_list)
    
    def _normalize_part_id(self, part_id: str) -> str:
        """Normalize part ID for matching."""
        # Remove common separators and lowercase
        normalized = re.sub(r'[\.\)\s\-]', '', part_id.lower())
        return normalized
