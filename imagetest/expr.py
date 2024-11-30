import logging
from typing import List

from configs import Configs
from interface import ExprMetrics, ExprRes

from vllm import LLM, RequestOutput, SamplingParams, TokensPrompt

# from vllm.global_metrics import GlobalMetrics, g_metrics
from vllm.sequence import RequestMetrics

logger = logging.getLogger(__name__)

def get_llm(configs: Configs) -> LLM:
	return LLM(
		model=configs.model,
		tensor_parallel_size=1,
		pipeline_parallel_size=1,
		dtype="half",
		max_model_len=32768,
		disable_async_output_proc=True,
		gpu_memory_utilization=0.8,
	)

def get_sp_template(configs: Configs) -> SamplingParams:
	return SamplingParams(
		temperature=0.8,
		top_p=0.95,
		max_tokens=1,
	)

def do_front_expr(configs: Configs, llm: LLM, len: int) -> RequestMetrics:
	# g_metrics.reset()
	# g_metrics.start = True
	sampling_params = get_sp_template(configs)
	prompt = TokensPrompt(prompt_token_ids=[10000] * len)
	outputs: List[RequestOutput] = llm.generate(prompt, sampling_params)
	return outputs[0].metrics

def do_end_expr(configs: Configs, llm: LLM, len: int) -> RequestMetrics:
	# g_metrics.reset()
	sampling_params = get_sp_template(configs)
	sampling_params.is_generate_cache = True
	prompt = TokensPrompt(prompt_token_ids=[10000] * (Configs.TOTAL_LEN - len))
	outputs: List[RequestOutput] = llm.generate(prompt, sampling_params)
	extra_sequence_group = outputs[0].seq_group

	# g_metrics.reset()
	# g_metrics.start = True
	sampling_params = get_sp_template(configs)
	sampling_params.extra_seq_groups = [extra_sequence_group]
	sampling_params.expire_seq_groups = [extra_sequence_group]
	sampling_params.recomp_block_num = 0
	prompt = TokensPrompt(prompt_token_ids=[10000] * len)
	outputs: List[RequestOutput] = llm.generate(prompt, sampling_params)
	return outputs[0].metrics

APPROACH_TO_TEST_FUNC = {
	"front": do_front_expr,
	"end": do_end_expr,
}
def run_single_expr(configs: Configs, llm: LLM, approach: str, len: int) -> RequestMetrics:
	return APPROACH_TO_TEST_FUNC[approach](configs, llm, len)

def run(configs: Configs):
	logger.info(f"Running {__name__}.run with configs {configs}")
	json_path = configs.get_result_filename("expr.json")
	res = ExprRes.from_json(json_path)
	# g_metrics.reset()
	llm = get_llm(configs)
	for approach in Configs.ALL_EXPR_APPROACHS:
		for length in Configs.ALL_LENS:
			metric_list = res.get_my(approach, length)
			total_times = 0
			while total_times < Configs.TOTAL_TIMES and len(metric_list) < Configs.TOTAL_TIMES:

				metrics = run_single_expr(configs, llm, approach, length)

				t1 = 0.0
				t2 = 0.0
				t3 = 0.0
				t4 = 0.0
				# for i in range(len(g_metrics.tp_1)):
				# 	try:
				# 		t1 += g_metrics.tp_1[i].elapsed_time(g_metrics.tp_2[i])
				# 	except Exception as e:
				# 		logger.warning(f"Error in elapsed_time: {e}")
				# 	try:
				# 		t2 += g_metrics.tp_2[i].elapsed_time(g_metrics.tp_3[i])
				# 	except Exception as e:
				# 		logger.warning(f"Error in elapsed_time: {e}")
				# 	try:
				# 		t3 += g_metrics.tp_3[i].elapsed_time(g_metrics.tp_4[i])
				# 	except Exception as e:
				# 		logger.warning(f"Error in elapsed_time: {e}")
				# 	try:
				# 		t4 += g_metrics.tp_4[i].elapsed_time(g_metrics.tp_5[i])
				# 	except Exception as e:
				# 		logger.warning(f"Error in elapsed_time: {e}")
				
				expr_metrics = ExprMetrics(
					arrival_time=metrics.arrival_time,
					last_token_time=metrics.last_token_time,
					first_scheduled_time=metrics.first_scheduled_time,
					first_token_time=metrics.first_token_time,
					time_in_queue=metrics.time_in_queue,
					finished_time=metrics.finished_time,
					scheduler_time=metrics.scheduler_time,
					model_forward_time=metrics.model_forward_time,
					model_execute_time=metrics.model_execute_time,
					input_layernorm_elapsed=t1 / 1000,
					self_attn_elapsed=t2 / 1000,
					post_attn_layernorm_elapsed=t3 / 1000,
					mlp_elapsed=t4 / 1000,
				)
				metric_list.append(expr_metrics)
				total_times += 1
			if len(metric_list) < Configs.TOTAL_TIMES:
				logger.warning(f"Only {len(metric_list)} metrics for approach {approach} len {len}")
			res.update_my(approach, length, metric_list)
	res.save_to_json(json_path)

if __name__ == "__main__":
	configs = Configs(model=Configs.ALL_MODELS[0])
	llm = get_llm(configs)
	m2 = run_single_expr(configs, llm, "end", 16)
	print(m2, m2.first_token_time - m2.first_scheduled_time)
	m1 = run_single_expr(configs, llm, "front", 16)
	print(m1, m1.first_token_time - m1.first_scheduled_time)