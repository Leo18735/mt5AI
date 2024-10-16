import os.path
import pickle


class Dumper:
    def __init__(self, path_name: str, dump: bool = True):
        self._path_name: str = path_name
        self._values = self.load()
        self._dump: bool = dump

    def add(self, result):
        self._values.append(result)
        if self._dump:
            self._store()

    def load(self):
        if not os.path.exists(self._path_name) or not self._dump:
            return []
        with open(self._path_name, "rb") as f:
            return pickle.load(f)

    def _store(self):
        with open(self._path_name, "wb") as f:
            pickle.dump(self._values, f)

    def exists(self, result) -> bool:
        for value in self._values:
            if value[0] == result:
                return True
        return False

    def get_results(self):
        return self._values
