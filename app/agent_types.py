# app/agent_types.py
from agents import Agent, Runner, ModelSettings, function_tool, handoff, trace
import uuid

def gen_trace_id() -> str:
    """Generate a unique trace ID in OpenAI trace format."""
    return f"trace_{uuid.uuid4().hex}"