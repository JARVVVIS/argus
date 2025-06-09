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
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np

import argparse

warnings.filterwarnings("ignore")


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, sample_fps, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames, frame_time, video_time


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def load_model(args):

    if args.model_name == "llava_video_7b":
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
    elif args.model_name == "llava_video_7b_video_only":
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2-Video-Only"
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
    elif args.model_name == "llava_next_video_7b_dpo":
        pretrained = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
        # model_name = get_model_name_from_path(pretrained)
        model_name = "llava_llama"
        print(f"Model name: {model_name}")
        device = "cuda"
        device_map = "auto"
    elif args.model_name == "llava_next_video_7b":
        pretrained = "lmms-lab/LLaVA-NeXT-Video-7B"
        model_name = get_model_name_from_path(pretrained)
        print(f"Model name: {model_name}")
        device = "cuda"
        device_map = "auto"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map="cuda"
    )
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "max_length": max_length,
        "device": device,
    }


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    tokenizer, model, image_processor, max_length, device = (
        model_dict["tokenizer"],
        model_dict["model"],
        model_dict["image_processor"],
        model_dict["max_length"],
        model_dict["device"],
    )

    video, frame_time, video_time = load_video(
        args.clip_path, args.num_frames, 1, force_sample=True
    )
    video = (
        image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda()
    ).to(torch.bfloat16)

    video = [video]
    conv_template = (
        "qwen_1_5"  # Make sure you use correct chat template for different models
    )
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    inp_prompt = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], inp_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava_video_7b",
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

    models_to_test = [
        "llava_video_7b",
        "llava_video_7b_video_only",
        "llava_next_video_7b_dpo",
        "llava_next_video_7b",
    ]
    if args.test_all_models:
        for model_name in models_to_test:
            args.model_name = model_name
            model_dict = load_model(args)
            caption = generate_caption(args, model_dict)
            print(f"Model: {model_name}")
            print(f"Caption: {caption} [Type: {type(caption)}]")
            print("-" * 50)
    else:
        model_dict = load_model(args)
        caption = generate_caption(args, model_dict)
        print(caption)


if __name__ == "__main__":
    main()
