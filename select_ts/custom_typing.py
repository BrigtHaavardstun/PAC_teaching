from typing import Any, Callable, List, Protocol, Dict


__all__ = ['GetLabel', 'GetError', 'SimFunc', 'HuristicFunc']


class GetLabel(Protocol):
    """
    Return the label of concept c on example x.
    """

    def __call__(self, c: int, x: int) -> bool:
        ...


class GetError(Protocol):
    """
    Return error probability (range 0-1) for concept c on example x.
    """

    def __call__(self, c: int, x: int) -> float:
        ...


class SimFunc(Protocol):
    """
    Similarity function
    """

    def __call__(self, c: int, c_target: int) -> float:
        ...


class HuristicFunc(Protocol):
    """
    Huristic function
    """

    def __call__(self, c_target: int, X: List[int], C: List[int], G: List[int], B: List[int], label: GetLabel, err: GetError) -> Dict[int, float]:
        ...
