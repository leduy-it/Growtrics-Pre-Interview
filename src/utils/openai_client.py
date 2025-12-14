"""OpenAI API client wrapper with logging and cost tracking."""

import base64
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config.settings import get_settings
from .cost_tracker import CostTracker
from .logger import AgentLogger


class OpenAIClient:
    """Wrapper for OpenAI API with built-in logging and cost tracking."""
    
    def __init__(
        self,
        output_dir: str,
        cost_tracker: Optional[CostTracker] = None,
        model_name: Optional[str] = None
    ):
        settings = get_settings()
        
        # Configure API
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.model_name = model_name or settings.OPENAI_MODEL
        self.output_dir = output_dir
        self.logger = AgentLogger(output_dir)
        self.cost_tracker = cost_tracker or CostTracker()
    
    async def generate_with_image(
        self,
        prompt: str,
        images: List[bytes],
        agent_name: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate response with image input using GPT-4 Vision."""
        start_time = time.time()
        
        # Build messages with images
        content = []
        
        # Add images as base64
        for img_bytes in images:
            base64_image = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})
        
        # Log input
        self.logger.log_input(
            agent_name=agent_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "data": f"[{len(images)} images]"},
                    {"type": "text", "text": prompt}
                ]}
            ],
            metadata={"system_instruction": system_instruction, "model": self.model_name}
        )
        
        try:
            vision_model = self.model_name
            
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_completion_tokens=4096,
                temperature=0.1,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Track cost
            cost = self.cost_tracker.track_call(
                agent_name=agent_name,
                model=vision_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms
            )
            
            response_text = response.choices[0].message.content
            
            # Log output
            self.logger.log_output(
                agent_name=agent_name,
                messages=[{"role": "assistant", "content": response_text}],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms
            )
            
            return {
                "text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            self.logger.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_text(
        self,
        prompt: str,
        agent_name: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate text-only response using OpenAI."""
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        # Log input
        self.logger.log_input(
            agent_name=agent_name,
            messages=messages,
            metadata={"model": self.model_name}
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract token counts
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Track cost
            cost = self.cost_tracker.track_call(
                agent_name=agent_name,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms
            )
            
            response_text = response.choices[0].message.content
            
            # Log output
            self.logger.log_output(
                agent_name=agent_name,
                messages=[{"role": "assistant", "content": response_text}],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_ms=duration_ms
            )
            
            return {
                "text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            self.logger.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def detect_bounding_boxes(
        self,
        image: bytes,
        prompt: str,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Detect bounding boxes using GPT-4 Vision."""
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
