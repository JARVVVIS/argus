import os
import argparse
import pandas as pd
import tqdm
from datasets import load_dataset

from PIL import Image

import torch
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu  # pip install decord

MAX_NUM_FRAMES = 16


def encode_video(video_path, num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    assert (
        num_frames <= MAX_NUM_FRAMES
    ), f"num_frames should be less than or equal to {MAX_NUM_FRAMES}"

    frame_idx = uniform_sample(frame_idx, num_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    print("num frames:", len(frames))
    return frames


def load_model(args):

    if args.model_name == "mplug_owl3_7b":
        pretrained = "mPLUG/mPLUG-Owl3-7B-241101"
    elif args.model_name == "mplug_owl3_1b":
        pretrained = "mPLUG/mPLUG-Owl3-1B-241014"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        pretrained,
        attn_implementation="sdpa",
        torch_dtype=torch.half,
        trust_remote_code=True,
    )
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    processor = model.init_processor(tokenizer)

    return {"model": model, "tokenizer": tokenizer, "processor": processor}


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    model, tokenizer, processor = (
        model_dict["model"],
        model_dict["tokenizer"],
        model_dict["processor"],
    )

    messages = [
        {
            "role": "user",
            "content": f"""<|video|>
        {prompt}.""",
        },
        {"role": "assistant", "content": ""},
    ]

    videos = [args.clip_path]

    video_frames = [encode_video(_, args.num_frames) for _ in videos]

    inputs = processor(messages, images=None, videos=video_frames)

    inputs.to("cuda")
    inputs.update(
        {
            "tokenizer": tokenizer,
            "max_new_tokens": 1000,
            "decode_text": True,
        }
    )

    try:
        g = model.generate(**inputs)
        response = g[0]
    except Exception as e:
        print(f"Error: {e}")
        response = None

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="mplug_owl3_7b",
    )
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon.mp4",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    args.clip_path = args.clip_name

    model_dict = load_model(args)
    caption = generate_caption(args, model_dict)
    print(f"Caption:\n{caption}")


if __name__ == "__main__":
    main()
