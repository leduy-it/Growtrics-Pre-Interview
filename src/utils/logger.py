"""Structured logging for agent interactions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.logging_schema import AgentLog, AgentMessage


class AgentLogger:
    """Logger for capturing agent input/output with image omission."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir) / "logs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger("ocr_pipeline")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def log_input(
        self,
        agent_name: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log agent input messages."""
        return self._log_interaction(agent_name, "input", messages, metadata)
    
    def log_output(
        self,
        agent_name: str,
        messages: List[Dict[str, Any]],
        tokens: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log agent output messages with token/cost info."""
        return self._log_interaction(
            agent_name, 
            "output", 
            messages, 
            metadata,
            tokens=tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms
        )
    
    def _log_interaction(
        self,
        agent_name: str,
        direction: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Internal method to log an interaction."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{agent_name}_{direction}.log"
        filepath = self.output_dir / filename
        
        # Convert to AgentMessage objects
        agent_messages = []
        for msg in messages:
            agent_messages.append(AgentMessage(
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
                timestamp=datetime.now()
            ))
        
        # Create log entry
        log_entry = AgentLog(
            agent_name=agent_name,
            direction=direction,
            messages=agent_messages,
            **kwargs
        )
        
        # Write to file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(log_entry.to_log_string())
            if metadata:
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("Metadata:\n")
                f.write("=" * 50 + "\n")
                f.write(json.dumps(metadata, indent=2, default=str))
        
        self.logger.info(f"Logged {direction} for {agent_name}: {filename}")
        return str(filepath)


def log_agent_interaction(
    output_dir: str,
    agent_name: str,
    direction: str,
    messages: List[Dict[str, Any]],
    **kwargs
) -> str:
    """Convenience function to log a single interaction."""
    logger = AgentLogger(output_dir)
    if direction == "input":
        return logger.log_input(agent_name, messages)
    else:
        return logger.log_output(agent_name, messages, **kwargs)
