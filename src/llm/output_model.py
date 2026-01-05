from typing import Dict, Optional

from pydantic import BaseModel, Field


class Action(BaseModel):
    """
    Executable action with its argument.

    Parameters
    ----------
    type : str
        Type of action to execute, such as 'move' or 'speak'
    value : str
        The action argument, such as the magnitude of a movement or the sentence to speak
    args : dict, optional
        All function call arguments for actions with multiple parameters.
        When present, the orchestrator will pass all args to the action interface.
    """

    type: str = Field(
        ..., description="The specific type of action, such as 'move' or 'speak'"
    )
    value: str = Field(..., description="The action argument")
    args: Optional[Dict[str, str]] = Field(
        default=None,
        description="All function call arguments for multi-parameter actions",
    )


class CortexOutputModel(BaseModel):
    """
    Output model for the Cortex LLM responses.

    Parameters
    ----------
    actions : list[Action]
        List of actions to be executed
    """

    actions: list[Action] = Field(..., description="List of actions to execute")
