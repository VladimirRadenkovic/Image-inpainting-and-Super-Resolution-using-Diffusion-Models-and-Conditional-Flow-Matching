from typing import Sequence

class KeepFields:
    def __init__(self, fields_to_keep: Sequence[str]):
        self.fields_to_keep = fields_to_keep

    def __call__(self, data_dict: dict) -> dict:
        data_dict = {k: v for k, v in data_dict.items() if k in self.fields_to_keep}
        return data_dict