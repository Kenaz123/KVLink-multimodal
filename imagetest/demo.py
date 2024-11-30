import logging
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import torch
from configs import Configs
from interface import ExprMetrics, ExprRes

from vllm import LLM, RequestOutput, SamplingParams, TokensPrompt
from vllm.global_metrics import GlobalMetrics, g_metrics
from vllm.sequence import RequestMetrics

logger = logging.getLogger(__name__)
def get_llm(configs: Configs) -> LLM:
	return LLM(
		model=configs.model,
		tensor_parallel_size=1,
		pipeline_parallel_size=1,
		dtype="half",
		max_model_len=32768,
		disable_async_output_proc=False,
		gpu_memory_utilization=0.8,
	)

def get_sp_template(configs: Configs) -> SamplingParams:
	return SamplingParams(
		max_tokens=1,
	)

def do_front_expr(configs: Configs, llm: LLM, len: int) -> RequestMetrics:
	sampling_params = get_sp_template(configs)
	prompt = TokensPrompt(prompt_token_ids=[10000] * len)
	outputs: List[RequestOutput] = llm.generate(prompt, sampling_params)
	return outputs[0].metrics

if __name__ == '__main__':
	configs = Configs(model=Configs.ALL_MODELS[0])
	llm = get_llm(configs)

	g_metrics.start = True
	metrics = do_front_expr(configs, llm, 30000)
	g_metrics.arrival_time = metrics.arrival_time
	g_metrics.first_scheduled_time = metrics.first_scheduled_time
	g_metrics.first_token_time = metrics.first_token_time

	pprint({
		"first_scheduled_time": g_metrics.first_scheduled_time - g_metrics.arrival_time,
		"first_token_time": g_metrics.first_token_time - g_metrics.arrival_time,
	})

	torch.cuda.synchronize()
	assert len(g_metrics.tp_1) == len(g_metrics.tp_2) == len(g_metrics.tp_3) == len(g_metrics.tp_4) == len(g_metrics.tp_5)
	t1 = 0.0
	t2 = 0.0
	t3 = 0.0
	t4 = 0.0
	for i in range(len(g_metrics.tp_1)):
		try:
			t1 += g_metrics.tp_1[i].elapsed_time(g_metrics.tp_2[i])
		except Exception as e:
			logger.warning(f"Error in elapsed_time: {e}")
		try:
			t2 += g_metrics.tp_2[i].elapsed_time(g_metrics.tp_3[i])
		except Exception as e:
			logger.warning(f"Error in elapsed_time: {e}")
		try:
			t3 += g_metrics.tp_3[i].elapsed_time(g_metrics.tp_4[i])
		except Exception as e:
			logger.warning(f"Error in elapsed_time: {e}")
		try:
			t4 += g_metrics.tp_4[i].elapsed_time(g_metrics.tp_5[i])
		except Exception as e:
			logger.warning(f"Error in elapsed_time: {e}")

	pprint({
		"1_input_layernorm_ms": t1,
		"2_self_attn_ms": t2,
		"3_post_attn_layernorm_ms": t3,
		"4_mlp_ms": t4,
	})