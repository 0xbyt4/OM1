from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class HeadAction(str, Enum):
    """
    Enumeration of possible head movements.
    """

    LOOK_LEFT = "look left"
    LOOK_RIGHT = "look right"
    LOOK_UP = "look up"
    LOOK_DOWN = "look down"
    LOOK_AT_PERSON = "look at person"
    CENTER = "center"


@dataclass
class HeadInput:
    """
    Input interface for the HeadControl action.
    """

    action: HeadAction


@dataclass
class HeadControl(Interface[HeadInput, HeadInput]):
    """
    This action allows you to control head movement and gaze direction.
    """

    input: HeadInput
    output: HeadInput
