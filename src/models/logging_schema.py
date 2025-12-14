"""Pydantic schemas for structured logging of agent interactions."""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import re


class AgentMessage(BaseModel):
    """A single message in an agent conversation."""
    role: str = Field(..., description="Message role: 'user', 'assistant', 'system'")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Text or multimodal content")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def to_log_format(self) -> Dict[str, Any]:
        """Convert to log format with image bytes omitted."""
        content = self.content
        
        if isinstance(content, list):
            # Handle multimodal content
            processed_content = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = self._process_content_item(item)
                    processed_content.append(processed_item)
                else:
                    processed_content.append(item)
            content = processed_content
        elif isinstance(content, str):
            # Check for embedded base64 images
            content = self._omit_base64_images(content)
        
        return {
            "role": self.role,
            "content": content,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def _process_content_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single content item, omitting image bytes."""
        processed = {}
        for key, value in item.items():
            if key in ("image", "image_url", "inline_data"):
                if isinstance(value, dict) and "data" in value:
                    size = len(value.get("data", ""))
                    processed[key] = f"[image: size {size} bytes; omitted]"
                elif isinstance(value, bytes):
                    processed[key] = f"[image: size {len(value)} bytes; omitted]"
                elif isinstance(value, str) and len(value) > 1000:
                    # Likely base64
                    processed[key] = f"[image: size {len(value)} bytes; omitted]"
                else:
                    processed[key] = value
            elif isinstance(value, bytes):
                processed[key] = f"[binary: size {len(value)} bytes; omitted]"
            else:
                processed[key] = value
        return processed
    
    def _omit_base64_images(self, text: str) -> str:
        """Replace base64 image data in text with placeholder."""
        # Match base64 image patterns
        pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}'
        
        def replacer(match):
            size = len(match.group(0))
            return f"[image: size {size} bytes; omitted]"
        
        return re.sub(pattern, replacer, text)


class AgentLog(BaseModel):
    """Complete log for an agent interaction."""
    agent_name: str = Field(..., description="Name of the agent")
    direction: str = Field(..., description="'input' or 'output'")
    messages: List[AgentMessage] = Field(default_factory=list)
    tokens_used: Optional[int] = Field(None, description="Total tokens used")
    input_tokens: Optional[int] = Field(None)
    output_tokens: Optional[int] = Field(None)
    cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")
    duration_ms: Optional[float] = Field(None, description="Processing time in ms")
    
    def to_log_string(self) -> str:
        """Convert to formatted log string."""
        lines = [
            f"Agent: {self.agent_name}",
            f"Direction: {self.direction}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
        ]
        
        if self.tokens_used:
            lines.append(f"Tokens Used: {self.tokens_used}")
        if self.input_tokens:
            lines.append(f"Input Tokens: {self.input_tokens}")
        if self.output_tokens:
            lines.append(f"Output Tokens: {self.output_tokens}")
        if self.cost_usd:
            lines.append(f"Estimated Cost: ${self.cost_usd:.6f}")
        if self.duration_ms:
            lines.append(f"Duration: {self.duration_ms:.2f}ms")
        
        lines.append("")
        lines.append("=" * 50)
        lines.append("Messages:")
        lines.append("=" * 50)
        
        for i, msg in enumerate(self.messages):
            lines.append(f"\n--- Message {i + 1} ({msg.role}) ---")
            log_content = msg.to_log_format()
            lines.append(json.dumps(log_content, indent=2, default=str))
        
        return "\n".join(lines)
