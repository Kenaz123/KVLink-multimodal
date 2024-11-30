"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""
from argparse import Namespace
from typing import List

from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

QUESTION = "What is the content of each image?"
# QUESTION = "Which are the images containing the animals?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
    # "images/cherry_blossom.jpg",
    # "images/stop_sign.jpg",
    # "images/5.jpg",
    # "images/13.jpg",
    # "images/14.jpg",
]

def load_qwenvl_chat(question: str, image_urls: List[str]):
    model_name = "Qwen/Qwen-VL-Chat"
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": len(image_urls)},
        disable_async_output_proc=True,
    )
    placeholders = "".join(f"Picture {i}: <img></img>\n"
                           for i, _ in enumerate(image_urls, start=1))

    # This model does not have a chat_template attribute on its tokenizer,
    # so we need to explicitly pass it. We use ChatML since it's used in the
    # generation utils of the model:
    # https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/qwen_generation_utils.py#L265
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    # Copied from: https://huggingface.co/docs/transformers/main/en/chat_templating
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501

    messages = [{'role': 'user', 'content': f"{placeholders}\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=chat_template)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids, None, chat_template

model_example_map = {
    "qwen_vl_chat": load_qwenvl_chat,
}

def run_generate(model, question: str, image_urls: List[str]):
    llm, prompt, stop_token_ids, image_data, _ = model_example_map[model](
        question, image_urls)
    if image_data is None:
        image_data = [fetch_image(url) for url in image_urls]

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=stop_token_ids)
    sampling_params.is_generate_cache = True

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        },
        sampling_params=sampling_params)
    extra_sequence_group = outputs[0].seq_group
    sampling_params.extra_seq_groups = [extra_sequence_group]
    sampling_params.expire_seq_groups = [extra_sequence_group]
    sampling_params.recomp_block_num = 20
    model_name = "Qwen/Qwen-VL-Chat"
    placeholders = "".join(f"Picture {i}: <img></img>\n"
                           for i, _ in enumerate(image_urls, start=1))
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)

    # Copied from: https://huggingface.co/docs/transformers/main/en/chat_templating
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # noqa: E501
    question2 = "Which are the images containing the animals?"
    messages = [{'role': 'user', 'content': f"{placeholders}\n{question2}"}]
    new_prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True,
                                           chat_template=chat_template)
    print(new_prompt)
    new_output = llm.generate(
        {
            "prompt": new_prompt,
            # "multi_modal_data": {
            #     "image": image_data
            # },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    
    for o in new_output:
        generated_text = o.outputs[0].text
        print(generated_text)

def main(args: Namespace):
    model = args.model_type
    method = args.method

    if method == "generate":
        run_generate(model, QUESTION, IMAGE_URLS)
    else:
        raise ValueError(f"Invalid method: {method}")

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with '
        'vision language models that support multi-image input')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="qwen_vl_chat",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")

    args = parser.parse_args()
    main(args)