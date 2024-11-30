import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from configs import Configs


@dataclass
class ExprMetrics:
	arrival_time: float
	last_token_time: float
	first_scheduled_time: Optional[float]
	first_token_time: Optional[float]
	time_in_queue: Optional[float]
	finished_time: Optional[float] = None
	scheduler_time: Optional[float] = None
	model_forward_time: Optional[float] = None
	model_execute_time: Optional[float] = None
	input_layernorm_elapsed: Optional[float] = None
	self_attn_elapsed: Optional[float] = None
	post_attn_layernorm_elapsed: Optional[float] = None
	mlp_elapsed: Optional[float] = None

	def to_dict(self) -> Dict:
		return {
            "arrival_time": self.arrival_time,
			"last_token_time": self.last_token_time,
			"first_scheduled_time": self.first_scheduled_time,
			"first_token_time": self.first_token_time,
			"time_in_queue": self.time_in_queue,
			"finished_time": self.finished_time,
			"scheduler_time": self.scheduler_time,
			"model_forward_time": self.model_forward_time,
			"model_execute_time": self.model_execute_time,
			"input_layernorm_elapsed": self.input_layernorm_elapsed,
			"self_attn_elapsed": self.self_attn_elapsed,
			"post_attn_layernorm_elapsed": self.post_attn_layernorm_elapsed,
			"mlp_elapsed": self.mlp_elapsed
		}
	
	@staticmethod
	def from_dict(d: Dict) -> "ExprMetrics":
		return ExprMetrics(
			arrival_time=d["arrival_time"],
			last_token_time=d["last_token_time"],
			first_scheduled_time=d["first_scheduled_time"],
			first_token_time=d["first_token_time"],
			time_in_queue=d["time_in_queue"],
			finished_time=d["finished_time"],
			scheduler_time=d["scheduler_time"],
			model_forward_time=d["model_forward_time"],
			model_execute_time=d["model_execute_time"],
			input_layernorm_elapsed=d["input_layernorm_elapsed"],
			self_attn_elapsed=d["self_attn_elapsed"],
			post_attn_layernorm_elapsed=d["post_attn_layernorm_elapsed"],
			mlp_elapsed=d["mlp_elapsed"]
		)

def _verify_id(approach: str, len: int) -> None:
	if approach not in Configs.ALL_EXPR_APPROACHS:
		raise ValueError(f"Invalid approach: {approach}")
	if len not in Configs.ALL_LENS:
		raise ValueError(f"Invalid len: {len}")

class ExprRes(Dict[str, List[ExprMetrics]]):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def to_dict(self) -> Dict:
		return {k: [s.to_dict() for s in v] for k, v in self.items()}
	
	@staticmethod
	def from_dict(d: Dict) -> "ExprRes":
		return ExprRes({k: [ExprMetrics.from_dict(s) for s in v] for k, v in d.items()})

	@staticmethod
	def from_json(filepath: str) -> "ExprRes":
		if os.path.exists(filepath):
			with open(filepath, "r") as f:
				return ExprRes.from_dict(json.load(f))
		else:
			return ExprRes()

	def save_to_json(self, filepath: str) -> None:
		with open(filepath, "w") as f:
			json.dump(self.to_dict(), f)
	
	@staticmethod
	def form_id(approach: str, len: int) -> str:
		_verify_id(approach, len)
		return f"{approach}_{len}_{Configs.TOTAL_LEN}"

	def update_my(self, approach: str, len: int, metrics_list: List[ExprMetrics]) -> None:
		self[self.form_id(approach, len)] = metrics_list

	def get_my(self, approach: str, len: int) -> List[ExprMetrics]:
		return self.get(self.form_id(approach, len), [])