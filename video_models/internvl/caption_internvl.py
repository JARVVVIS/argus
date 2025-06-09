import os
import argparse
import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from datasets import load_dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_model(args):
    if args.model_name == "internvl2_1b":
        pretrained = "OpenGVLab/InternVL2-1B"
    elif args.model_name == "internvl2_2b":
        pretrained = "OpenGVLab/InternVL2-2B"
    elif args.model_name == "internvl2_4b":
        pretrained = "OpenGVLab/InternVL2-4B"
    elif args.model_name == "internvl2_8b":
        pretrained = "OpenGVLab/InternVL2-8B"
    elif args.model_name == "internvl2_26b":
        pretrained = "OpenGVLab/InternVL2-26B"
    elif args.model_name == "internvl2_40b":
        pretrained = "OpenGVLab/InternVL2-40B"
    elif args.model_name == "internvl2_76b":
        pretrained = "OpenGVLab/InternVL2-Llama3-76B"

    elif args.model_name == "internvl2_5_1b":
        pretrained = "OpenGVLab/InternVL2_5-1B"
    elif args.model_name == "internvl2_5_2b":
        pretrained = "OpenGVLab/InternVL2_5-2B"
    elif args.model_name == "internvl2_5_4b":
        pretrained = "OpenGVLab/InternVL2_5-4B"
    elif args.model_name == "internvl2_5_8b":
        pretrained = "OpenGVLab/InternVL2_5-8B"
    elif args.model_name == "internvl2_5_26b":
        pretrained = "OpenGVLab/InternVL2_5-26B"
    elif args.model_name == "internvl2_5_38b":
        pretrained = "OpenGVLab/InternVL2_5-38B"
    elif args.model_name == "internvl2_5_76b":
        pretrained = "OpenGVLab/InternVL2_5-Llama3-76B"

    elif args.model_name == "internvl2_5_chat_8b":
        pretrained = "OpenGVLab/InternVideo2_5_Chat_8B"
    elif args.model_name == "internvl2_5_hi_co_r16":
        pretrained = "OpenGVLab/InternVL_2_5_HiCo_R16"
    elif args.model_name == "internvl2_5_hi_co_r64":
        pretrained = "OpenGVLab/InternVL_2_5_HiCo_R64"

    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    model = (
        AutoModel.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained, trust_remote_code=True, use_fast=False
    )
    return {"model": model, "tokenizer": tokenizer}


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    model, tokenizer = model_dict["model"], model_dict["tokenizer"]

    generation_config = dict(max_new_tokens=1024, do_sample=True)
    pixel_values, num_patches_list = load_video(
        args.clip_path, num_segments=args.num_frames, max_num=1
    )
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = "".join(
        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
    )
    print(f"Video prefix:\n{video_prefix}")
    inp_prompt = video_prefix + prompt

    response, history = model.chat(
        tokenizer,
        pixel_values,
        inp_prompt,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,  # none to ensure no leakage
        return_history=True,
    )

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="internvl2_8b",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--test_all_models", action="store_true", default=False)
    parser.add_argument(
        "--prompt", type=str, default="Describe the video in great detail"
    )
    args = parser.parse_args()

    args.clip_path = args.clip_name

    models_to_test = [
        "internvl2_1b",
        "internvl2_2b",
        "internvl2_4b",
        "internvl2_8b",
        "internvl2_26b",
        "internvl2_40b",
        "internvl2_76b",
        "internvl2_5_1b",
        "internvl2_5_2b",
        "internvl2_5_4b",
        "internvl2_5_8b",
        "internvl2_5_26b",
        "internvl2_5_38b",
        "internvl2_5_76b",
        "internvl2_5_chat_8b",
        "internvl2_5_hi_co_r16",
        "internvl2_5_hi_co_r64",
    ]

    model_dict = load_model(args)

    prompt = (
        "Does the chamaeleon change color in this video? (Answer only with yes or no)"
    )

    prompt = "How many chameleons are in this video? Answer with one of the following options: (A) One, (B) Two, (C) Three"

    generate_caption(args, model_dict, prompt=prompt)

    if args.test_all_models:
        for model_name in models_to_test:
            args.model_name = model_name

            model_dict = load_model(args)
            caption = generate_caption(args, model_dict, prompt=args.prompt)
            print(f"Model: {args.model_name}")
            print(f"Output: {caption}")
    else:
        model_dict = load_model(args)
        caption = generate_caption(args, model_dict, prompt=args.prompt)
        print(f"Model: {args.model_name}")
        print(f"Output: {caption}")


if __name__ == "__main__":
    main()
