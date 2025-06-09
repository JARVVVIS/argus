import sys
import glob
import argparse

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

import os
import tqdm
from datasets import load_dataset

import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os


def tensor_to_images(tensor, vid_name, output_folder="assets/sanity_check_videollama2"):
    # Ensure the output folder exists

    output_folder = f"{output_folder}/{vid_name}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Un-normalize using mean and std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    # Denormalize and convert to range 0-255
    tensor = tensor * std + mean  # un-normalize
    tensor = tensor.clamp(0, 1)  # clamp to the range [0, 1]

    # Convert tensor to PIL images and save
    for i in range(tensor.size(0)):
        img = to_pil_image(tensor[i])
        img.save(os.path.join(output_folder, f"{i}.jpg"))


def single_inference(
    model,
    tokenizer,
    processor,
    instruct,
    modal_path,
    modal="video",
    do_sample=False,
):

    vid_tensors = processor[modal](modal_path)
    vid_name = modal_path.split("/")[-1].split(".")[0]

    # tensor_to_images(vid_tensors, vid_name=vid_name)

    output = mm_infer(
        vid_tensors,
        instruct,
        model=model,
        tokenizer=tokenizer,
        do_sample=do_sample,
        modal=modal,
    )

    print("-" * 10)
    print(f"Video from Path: {modal_path}")
    print(f"Prompt: {instruct}")
    print(f"Model Output: {output}")
    print("-" * 10)

    return output


def load_model(args):
    if args.model_name == "videollama2_7b":
        pretrained = "DAMO-NLP-SG/VideoLLaMA2-7B"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    model, processor, tokenizer = model_init(pretrained, cache_dir="./cache")
    return {"model": model, "processor": processor, "tokenizer": tokenizer}


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    model, processor, tokenizer = (
        model_dict["model"],
        model_dict["processor"],
        model_dict["tokenizer"],
    )

    output = single_inference(
        model,
        tokenizer,
        processor,
        prompt,
        args.clip_path,
    )
    return output


def inference():
    disable_torch_init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="videollama2_7b")
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    args = parser.parse_args()

    args.clip_path = args.clip_name
    model_dict = load_model(args)
    output = generate_caption(args, model_dict)
    print(output)


if __name__ == "__main__":
    inference()
