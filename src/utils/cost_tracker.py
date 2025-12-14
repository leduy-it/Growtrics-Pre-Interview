"""Cost tracking for API calls."""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import json
from pathlib import Path


@dataclass
class APICall:
    """Record of a single API call."""
    timestamp: datetime
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: float


@dataclass
class CostTracker:
    """Track API costs and token usage across the pipeline."""
    
    PRICING: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gemini-flash-latest": {"input": 0.30, "output": 2.50},
        "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
        "gpt-4.1": {"input": 3.00, "output": 12.00},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5.2": {"input": 1.75, "output": 14.00},
    })
    
    calls: List[APICall] = field(default_factory=list)
    
    def track_call(
        self,
        agent_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float = 0.0
    ) -> float:
        """Track a single API call and return the cost."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        call = APICall(
            timestamp=datetime.now(),
            agent_name=agent_name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_ms=duration_ms
        )
        self.calls.append(call)
        
        return cost
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        pricing = self.PRICING.get(model, self.PRICING["gemini-1.5-flash"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict:
        """Get summary of all API calls."""
        if not self.calls:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "by_agent": {},
            }
        
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost_usd for c in self.calls)
        total_duration = sum(c.duration_ms for c in self.calls)
        
        # Group by agent
        by_agent = {}
        for call in self.calls:
            if call.agent_name not in by_agent:
                by_agent[call.agent_name] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_agent[call.agent_name]["calls"] += 1
            by_agent[call.agent_name]["input_tokens"] += call.input_tokens
            by_agent[call.agent_name]["output_tokens"] += call.output_tokens
            by_agent[call.agent_name]["cost_usd"] += call.cost_usd
        
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": total_cost,
            "total_duration_ms": total_duration,
            "by_agent": by_agent,
        }
    
    def save_report(self, output_dir: str) -> str:
        """Save cost report to file."""
        output_path = Path(output_dir) / "cost_report.json"
        
        report = {
            "summary": self.get_summary(),
            "calls": [
                {
                    "timestamp": c.timestamp.isoformat(),
                    "agent_name": c.agent_name,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                    "duration_ms": c.duration_ms,
                }
                for c in self.calls
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return str(output_path)
    
    def print_summary(self):
        """Print a formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("COST TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
        print(f"Total Duration: {summary.get('total_duration_ms', 0):.2f}ms")
        
        if summary['by_agent']:
            print("\nBy Agent:")
            for agent, stats in summary['by_agent'].items():
                print(f"  {agent}:")
                print(f"    Calls: {stats['calls']}")
                print(f"    Tokens: {stats['input_tokens']:,} in / {stats['output_tokens']:,} out")
                print(f"    Cost: ${stats['cost_usd']:.6f}")
        
        print("=" * 60 + "\n")
