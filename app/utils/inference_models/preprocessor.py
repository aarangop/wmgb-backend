
from abc import ABC, abstractmethod


class InputPreprocessor(ABC):

    @abstractmethod
    def preprocess_input(self, input):
        NotImplemented
