"""Diagram extraction agent - locates and extracts diagrams using bounding boxes."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io

from ..models import Question, BoundingBox
from ..utils.gemini_client import GeminiClient


DIAGRAM_DETECTION_PROMPT = """Analyze this exam page and identify ALL diagrams, figures, images, graphs, or visual elements.

For each diagram/figure found:
1. Describe what the diagram shows
2. Identify which question it belongs to (if applicable)
3. Provide precise bounding box coordinates
4. **Visual Reasoning**: Explain WHY you selected this bounding box. E.g., "Contains the triangle for Question 3" or "Captures the graph labeled Fig 2".

Return your response as valid JSON:
{
    "diagrams": [
        {
            "description": "Triangle ABC with sides labeled",
            "question_number": "3",
            "part_id": "3a",
            "bounding_box": {
                "x_min": 0.1,
                "y_min": 0.3,
                "x_max": 0.5,
                "y_max": 0.6
            },
            "reasoning": "Located the triangle shape next to question 3a text."
        }
    ]
}

Coordinate system:
- x_min, y_min: Top-left corner (values 0.0 to 1.0)
- x_max, y_max: Bottom-right corner (values 0.0 to 1.0)
- Coordinates are normalized relative to image dimensions

Be precise with bounding boxes - include the entire diagram with a small margin.
If no diagrams are found, return: {"diagrams": []}"""


class DiagramExtractor:
    """Extracts diagrams from exam pages using bounding box detection."""
    
    AGENT_NAME = "diagram_extractor"
    
    def __init__(self, gemini_client: GeminiClient, output_dir: str):
        self.client = gemini_client
        self.images_dir = Path(output_dir) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    async def extract_diagrams(
        self,
        page_images: List[bytes],
        questions: List[Question]
    ) -> List[Question]:
        """
        Extract diagrams and update questions with diagram paths.
        
        Args:
            page_images: List of page images as bytes
            questions: List of detected questions
            
        Returns:
            Updated questions with diagram paths filled in
        """
        # Build a mapping of question/part to Question/QuestionPart objects
        part_map = {}
        for q in questions:
            for part in q.question_parts:
                part_map[part.part_id] = part
        
        for page_num, page_image in enumerate(page_images):
            # Detect diagrams on this page
            response = await self.client.generate_with_image(
                prompt=DIAGRAM_DETECTION_PROMPT,
                images=[page_image],
                agent_name=self.AGENT_NAME,
                system_instruction="You are an expert at detecting diagrams and figures in exam papers. Return precise bounding box coordinates."
            )
            
            diagrams = self._parse_diagram_response(response["text"], page_num + 1)
            
            # Extract and save each diagram
            for diagram in diagrams:
                bbox = diagram.get("bounding_box")
                if not bbox:
                    continue
                
                # Crop and save the diagram
                try:
                    diagram_path = self._crop_and_save_diagram(
                        page_image=page_image,
                        bbox=bbox,
                        page_num=page_num + 1,
                        part_id=diagram.get("part_id", "unknown")
                    )
                    
                    # Update the corresponding question part
                    part_id = diagram.get("part_id")
                    if part_id and part_id in part_map:
                        # Store relative path
                        rel_path = f"images/{Path(diagram_path).name}"
                        part_map[part_id].diagrams.append(rel_path)
                    
                except Exception as e:
                    print(f"Warning: Failed to extract diagram: {e}")
        
        return questions
    
    def _parse_diagram_response(self, response_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Parse Gemini response into diagram data."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            diagrams = data.get("diagrams", [])
            
            # Add page number to each diagram
            for diagram in diagrams:
                if "bounding_box" in diagram:
                    diagram["bounding_box"]["page_number"] = page_num
                
                # Log reasoning if available
                if "reasoning" in diagram:
                    print(f"   [REASONING] {diagram.get('description', 'Diagram')}: {diagram['reasoning']}")
            
            return diagrams
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse diagram response: {e}")
            return []
    
    def _crop_and_save_diagram(
        self,
        page_image: bytes,
        bbox: Dict[str, float],
        page_num: int,
        part_id: str
    ) -> str:
        """Crop diagram from page image and save to disk."""
        # Load image
        img = Image.open(io.BytesIO(page_image))
        width, height = img.size
        
        # Convert normalized coordinates to pixels
        x_min = int(bbox["x_min"] * width)
        y_min = int(bbox["y_min"] * height)
        x_max = int(bbox["x_max"] * width)
        y_max = int(bbox["y_max"] * height)
        
        # Add small margin
        margin = 5
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        
        # Crop
        cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # Generate filename
        safe_part_id = re.sub(r'[^\w]', '_', part_id)
        filename = f"p{page_num}_{safe_part_id}_diagram.png"
        filepath = self.images_dir / filename
        
        # Save
        cropped.save(filepath, "PNG")
        
        return str(filepath)
