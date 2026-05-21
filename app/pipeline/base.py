from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class PipelineStep(ABC, Generic[I, O]):
    @abstractmethod
    async def run(self, input: I) -> O:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
