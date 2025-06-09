import os
import tqdm
import argparse
from datasets import load_dataset
import pandas as pd

from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")


# Function to extract frames from video
def load_video(video_path, max_frames_num=16):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(
        0, total_frame_num - 1, max_frames_num, dtype=int
    )
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def load_model(args):

    ## Load the model
    if args.model_name == "llava_ov_7b":
        pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
        model_hf_name = "llava_qwen"
    elif args.model_name == "llava_ov_0_5b_chat":
        pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        model_hf_name = "llava_qwen"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    device = "cuda:1"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained,
        None,
        model_hf_name,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "device": device,
    }


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    video_frames = load_video(args.clip_path, args.num_frames)
    image_tensors = []
    frames = (
        model_dict["image_processor"]
        .preprocess(video_frames, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    image_tensors.append(frames)

    conv_template = "qwen_1_5"
    question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt_question,
            model_dict["tokenizer"],
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(model_dict["device"])
    )
    image_sizes = [frame.size for frame in video_frames]

    try:
        # Generate response
        cont = model_dict["model"].generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = model_dict["tokenizer"].batch_decode(
            cont, skip_special_tokens=True
        )
        output = text_outputs[0]
    except Exception as e:
        print(f"Error: {e}")
        output = None

    return output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava_ov_7b",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--test_all_models", action="store_true", default=False)
    args = parser.parse_args()

    args.clip_path = args.clip_name

    if args.test_all_models:
        model_names = ["llava_ov_7b", "llava_ov_0_5b_chat"]
        for model_name in model_names:
            args.model_name = model_name
            model_dict = load_model(args)
            caption = generate_caption(args, model_dict)
            print(f"Model: {model_name}")
            print(f"Caption: {caption}")
            print("-" * 50)
    else:
        model_dict = load_model(args)
        caption = generate_caption(args, model_dict)
        print(caption)


if __name__ == "__main__":
    main()
