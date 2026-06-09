from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict


class TypeEnum(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class TypeModel(BaseModel):
    type: TypeEnum


class FunctionModel(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, TypeModel]
    returns: TypeModel


class PromptModel(BaseModel):
    prompt: str
