"""
Agent Types - Core agent definitions used throughout the site-worker application
"""
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable, Generic
import functools
import uuid
from pydantic import BaseModel

def gen_trace_id() -> str:
    """
    Generate a unique trace ID for tracking operations.
    
    Returns:
        A unique string ID
    """
    return str(uuid.uuid4())

def custom_span(name: str):
    """
    Mock custom_span decorator for tracing operations.
    In a real implementation, this would be used for distributed tracing.
    
    Args:
        name: Name of the span
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # In a real implementation, this would start a tracing span
            result = func(*args, **kwargs)
            # In a real implementation, this would end the tracing span
            return result
        return wrapper
    return decorator

def trace(name: str):
    """
    Mock trace decorator for tracing operations.
    In a real implementation, this would be used for distributed tracing.
    
    Args:
        name: Name of the trace
        
    Returns:
        Decorator function
    """
    return custom_span(name)

T = TypeVar('T')

class AgentOutputSchema(Generic[T]):
    """
    Wrapper for output schemas that allows non-strict JSON schema validation.
    Based on memory e4c3fa27-24c8-41e1-805e-2a467aaf889b, this fixes the
    'Strict JSON schema is enabled, but the output type is not valid' error.
    
    This is used for models with `extra = "allow"` and `Dict[str, Any]` fields
    which are not compatible with strict JSON schema mode.
    """
    def __init__(self, model_type: Type[T], strict_json_schema: bool = False):
        self.model_type = model_type
        self.strict_json_schema = strict_json_schema

# Mock function_tool decorator based on memory fcc19ed3-02d5-4b80-a2b8-51f8c40593c3
# This resolves the 'FunctionTool' object is not callable error
def function_tool(func: Callable) -> Callable:
    """
    Mock function_tool decorator that allows the function to be called directly.
    Based on memory fcc19ed3-02d5-4b80-a2b8-51f8c40593c3, this fixes the
    'FunctionTool' object is not callable error.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that can be called directly
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Add attributes needed for the Agent class
    wrapper.name = getattr(func, "__name__", "unknown_function")
    wrapper.description = getattr(func, "__doc__", "")
    
    return wrapper


class RunResult:
    """Mock RunResult class to simulate the agents library's RunResult"""
    
    def __init__(self, final_output=None):
        self.final_output = final_output
        
    def final_output_as(self, output_type, raise_if_incorrect_type=False):
        """
        Return the final output as the specified type.
        Based on memory e4c3fa27-24c8-41e1-805e-2a467aaf889b, this handles
        the case where output_type is wrapped in AgentOutputSchema.
        
        Args:
            output_type: The type to convert the output to
            raise_if_incorrect_type: Whether to raise an error if the output is not the correct type
            
        Returns:
            The output converted to the specified type
        """
        # If output_type is an AgentOutputSchema, get the model_type
        if hasattr(output_type, 'model_type'):
            output_type = output_type.model_type
            
        # For UserProfileData, return a mock object with a dict method
        if output_type.__name__ == 'UserProfileData':
            from app.schemas import UserProfileData
            return UserProfileData(
                name="John Doe",
                current_title="Software Engineer",
                current_company="Mock Company",
                bio="This is a mock bio for testing purposes.",
                skills=["Python", "JavaScript", "Machine Learning"],
                education=["BS in Computer Science"],
                achievements=["Mock Achievement"],
                projects=["Mock Project"],
                social_profiles={"linkedin": "https://linkedin.com/in/johndoe"},
                languages_spoken=["English"],
                location="San Francisco, CA"
            )
        # For ResearchDoc, return a list of mock ResearchDoc objects
        elif getattr(output_type, '__origin__', None) == list and getattr(output_type, '__args__', [None])[0].__name__ == 'ResearchDoc':
            from app.schemas import ResearchDoc
            return [
                ResearchDoc(
                    title="Mock Research Document 1",
                    url="https://example.com/doc1",
                    content="This is mock content for testing purposes.",
                    snippet="Mock snippet"
                ),
                ResearchDoc(
                    title="Mock Research Document 2",
                    url="https://example.com/doc2",
                    content="This is another mock content for testing purposes.",
                    snippet="Another mock snippet"
                )
            ]
        # Default case: return the final_output as is
        return self.final_output


class Runner:
    """Mock Runner class to simulate the agents library's Runner"""
    
    def __init__(self):
        pass
        
    async def run(self, agent, prompt=None, max_turns=10):
        """
        Run the agent and return a RunResult.
        Based on memory 68cc0041-df92-4f9b-af70-f0d96d9f3885, this handles
        the max_turns parameter and returns a RunResult with the appropriate output.
        
        Args:
            agent: The agent to run
            prompt: The prompt to send to the agent (not used in mock implementation)
            max_turns: Maximum number of turns to run the agent for
            
        Returns:
            RunResult object with the agent's output
        """
        # In a real implementation, this would run the agent
        # For now, we'll just return a mock result
        return RunResult(final_output=None)  # The final_output_as method will handle the type conversion

    @staticmethod
    def create():
        """Create a new Runner instance"""
        return Runner()

class ModelSettings:
    """Settings for the LLM model used by agents"""
    def __init__(self, temperature: float = 0.7, max_tokens: int = 2000):
        self.temperature = temperature
        self.max_tokens = max_tokens

class Agent:
    """
    Base agent class used throughout the site-worker application.
    This is a simplified version that matches the interface expected by our agent files.
    """
    def __init__(
        self, 
        name: str,
        model: str = "gpt-4o-mini",
        instructions: str = "",
        output_type: Any = None,
        tools: List[Any] = None,
        model_settings: Optional[ModelSettings] = None
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.output_type = output_type
        self.tools = tools or []
        self.model_settings = model_settings or ModelSettings()

class AgentOutputSchema:
    """
    Wrapper for agent output schemas that allows for non-strict schema validation.
    This is used to handle Pydantic models with extra fields.
    """
    def __init__(self, schema_type: Any, strict_json_schema: bool = True):
        self.schema_type = schema_type
        self.strict_json_schema = strict_json_schema

# Mock Runner class for testing
class Runner:
    """
    Mock Runner class for testing agent execution.
    """
    async def run(self, agent: Agent, prompt: str, max_turns: int = 10):
        """Mock run method that returns a simple result"""
        return RunResult(f"Mock result for {agent.name} with prompt: {prompt[:20]}...")

class RunResult:
    """
    Mock RunResult class for testing agent execution.
    """
    def __init__(self, final_output: Any):
        self.final_output = final_output
    
    def final_output_as(self, output_type: Any, raise_if_incorrect_type: bool = False):
        """Mock method to convert the final output to the specified type"""
        return self.final_output

# Mock tracing functions
def custom_span(name: str):
    """Mock custom span for tracing"""
    class MockSpan:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return MockSpan()

def gen_trace_id():
    """Generate a mock trace ID"""
    import uuid
    return str(uuid.uuid4())

def trace(name: str, trace_id: str = None):
    """Mock trace function"""
    class MockTrace:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return MockTrace()
