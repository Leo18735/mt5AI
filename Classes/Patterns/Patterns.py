import abc
import pandas as pd


class Patterns:
    @staticmethod
    @abc.abstractmethod
    def get_pattern(rates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        pass
