from dataclasses import dataclass


@dataclass(eq=True)
class RandomParameters:
    name: str = "random"
