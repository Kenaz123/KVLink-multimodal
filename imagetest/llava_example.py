import argparse
import os
import subprocess

import torch

from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData

# The assets are located at `s3://air-example-data-2/vllm_opensource_llava/`.
# file_name = ["images/stop_sign_pixel_values.pt","images/cherry_blossom_pixel_values.pt"]
# image_num = 2

def run_llava_pixel_values():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
    )

    prompt = "<image>" * 576  + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    # images = torch.load("images/stop_sign_pixel_values.pt")
    # images = []
    # for i in range(image_num):
    #     images.append(torch.load(file_name[i]))
    images = torch.load("images/stop_sign_pixel_values.pt")

    samplingparams = SamplingParams(max_tokens=150)

    outputs = llm.generate(prompt,
                           sampling_params=samplingparams,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_llava_image_features():
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        image_input_type="image_features",
        image_token_id=32000,
        image_input_shape="1,576,1024",
        image_feature_size=576,
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    # This should be provided by another online or offline component.
    images = torch.load("images/stop_sign_image_features.pt")

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args):
    if args.type == "pixel_values":
        run_llava_pixel_values()
    else:
        run_llava_image_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Llava")
    parser.add_argument("--type",
                        type=str,
                        choices=["pixel_values", "image_features"],
                        default="pixel_values",
                        help="image input type")
    args = parser.parse_args()
    # Download from s3
    # s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
    # local_directory = "images"

    # Make sure the local directory exists or create it
    # os.makedirs(local_directory, exist_ok=True)

    # Use AWS CLI to sync the directory
    # subprocess.check_call(
    #     ["aws", "s3", "sync", s3_bucket_path, local_directory])
    main(args)
