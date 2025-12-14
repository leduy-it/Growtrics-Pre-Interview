"""Gemini API client wrapper with logging and cost tracking."""

import asyncio
import base64
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import google.generativeai as genai
from PIL import Image
import io

from ..config.settings import get_settings
from .cost_tracker import CostTracker
from .logger import AgentLogger


class GeminiClient:
    """Wrapper for Gemini API with built-in logging and cost tracking."""
    
    RATE_LIMIT_DELAY = 15
    
    def __init__(
        self,
        output_dir: str,
        cost_tracker: Optional[CostTracker] = None,
        model_name: Optional[str] = None
    ):
        settings = get_settings()
        
        # Configure API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        self.model_name = model_name or settings.GEMINI_MODEL
        self.model = genai.GenerativeModel(self.model_name)
        
        self.output_dir = output_dir
        self.logger = AgentLogger(output_dir)
        self.cost_tracker = cost_tracker or CostTracker()
        
        # Track last call time for rate limiting
        self._last_call_time = 0
    
    async def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.RATE_LIMIT_DELAY:
            wait_time = self.RATE_LIMIT_DELAY - elapsed
            print(f"   [RATE LIMIT] waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
        self._last_call_time = time.time()
    
    async def generate_with_image(
        self,
        prompt: str,
        images: List[bytes],
        agent_name: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response with image input."""
        # Apply rate limiting
        await self._rate_limit()
        
        start_time = time.time()
        
        # Build content parts
        content_parts = []
        
        # Add images
        for img_bytes in images:
            content_parts.append({
                "mime_type": "image/png",
                "data": base64.b64encode(img_bytes).decode("utf-8")
            })
        
        # Add text prompt
        content_parts.append(prompt)
        
        # Log input
        self.logger.log_input(
            agent_name=agent_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "data": f"[{len(images)} images]"},
                    {"type": "text", "text": prompt}
                ]}
            ],
            metadata={"system_instruction": system_instruction}
        )
        
        # Generate
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response = model.generate_content(content_parts)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            
            # Track cost
            cost = self.cost_tracker.track_call(
                agent_name=agent_name,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms
            )
            
            # Log output
            self.logger.log_output(
                agent_name=agent_name,
                messages=[{"role": "assistant", "content": response.text}],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms
            )
            
            return {
                "text": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            self.logger.logger.error(f"Gemini API error: {e}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        agent_name: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate text-only response."""
        start_time = time.time()
        
        # Log input
        self.logger.log_input(
            agent_name=agent_name,
            messages=[{"role": "user", "content": prompt}],
            metadata={"system_instruction": system_instruction}
        )
        
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response = model.generate_content(prompt)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            
            # Track cost
            cost = self.cost_tracker.track_call(
                agent_name=agent_name,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms
            )
            
            # Log output
            self.logger.log_output(
                agent_name=agent_name,
                messages=[{"role": "assistant", "content": response.text}],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms
            )
            
            return {
                "text": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            self.logger.logger.error(f"Gemini API error: {e}")
            raise
    
    async def detect_bounding_boxes(
        self,
        image: bytes,
        prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Detect bounding boxes using Gemini's spatial understanding."""
        bbox_prompt = f"""
{prompt}

Return bounding boxes in the following JSON format:
{{
    "diagrams": [
        {{
            "description": "brief description of the diagram",
            "bounding_box": {{
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 1.0,
                "y_max": 1.0
            }}
        }}
    ]
}}

Coordinates should be normalized (0.0 to 1.0) relative to the image dimensions.
x_min, y_min is the top-left corner; x_max, y_max is the bottom-right corner.
"""
        return await self.generate_with_image(
            prompt=bbox_prompt,
            images=[image],
            agent_name=agent_name,
        )
