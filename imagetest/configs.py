import json
import os
from typing import Dict, List, Optional, Tuple


class Configs:
	RESULT_DIR = "./results"

	# approaches

	ALL_EXPR_APPROACHS: List[str] = ["front", "end"]

	TOTAL_LEN: int = 2 ** 15

	ALL_LENS: List[int] = [2 ** i for i in range(4, 15)]

	ALL_MODELS = ["mistralai/Mistral-7B-Instruct-v0.3"]

	TOTAL_TIMES = 5

	def __init__(self, model: str):
		self.model = model
		self._verify_configs()
	
	def _verify_configs(self):
		if self.model not in self.ALL_MODELS:
			raise ValueError(f"Invalid model: {self.model}")

	def to_dict(self) -> Dict:
		return {
			"model": self.model
		}

	@staticmethod
	def from_dict(d: Dict) -> 'Configs':
		return Configs(
			model=d["model"]
		)


	def __repr__(self) -> str:
		return str(self.to_dict())

	@property
	def result_dirname(self) -> str:
		return f"{self.model.split('/')[-1]}"

	@property
	def result_dirpath(self) -> str:
		path = os.path.join(Configs.RESULT_DIR, self.result_dirname)
		if not os.path.exists(path):
			os.makedirs(path)
		return path

	def get_result_filename(self, body: str) -> str:
		return os.path.join(self.result_dirpath, f"{body}")

	def __eq__(self, other: 'Configs') -> bool:
		return self.to_dict() == other.to_dict()

if __name__ == "__main__":
	configs = Configs("mistralai/Mistral-7B-Instruct-v0.3")
	breakpoint()