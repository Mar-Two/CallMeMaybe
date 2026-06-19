from pydantic import BaseModel, Field, ConfigDict, model_validator
from enum import Enum
from typing import Dict
from typing_extensions import Self


class TypeEnum(Enum):
    """Enumerates the supported JSON schema types for function arguments."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NONE = "none"


class TypeModel(BaseModel):
    """Represents a function argument's accepted type.

    Attributes:
        type: The type that can be accepted (number, string, boolean, etc.).
    """

    type: TypeEnum


class FunctionModel(BaseModel):
    """Represents a callable function definition with its schema.

    Extra fields not defined in the schema are rejected during validation.

    Attributes:
        name: The function's identifier.
        description: A human-readable explanation of what the function does.
        parameters: Mapping of argument names to their type definitions.
        returns: The expected return type of the function.
    """

    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, TypeModel]
    returns: TypeModel

    @model_validator(mode='after')
    def check_name(self) -> Self:
        if not self.name.isidentifier():
            raise ValueError(f"\nError: ('{self.name}') has an invalid "
                             "function name")
        return self


class PromptModel(BaseModel):
    """Represents a natural-language prompt to be processed.

    Attributes:
        prompt: The original natural-language request from the user.
    """

    prompt: str
