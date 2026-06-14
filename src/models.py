from pydantic import BaseModel, Field, ConfigDict, model_validator
from enum import Enum
from typing import Dict
from typing import Self


class TypeEnum(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NONE = "none"


class TypeModel(BaseModel):
    type: TypeEnum


class FunctionModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, TypeModel]
    returns: TypeModel

    @model_validator(mode='after')
    def check_name(self) -> Self:
        if not self.name.isidentifier():
            raise ValueError(f"\nError: {self.name} has an invalid "
                             "function name")
        return self


class PromptModel(BaseModel):
    prompt: str
