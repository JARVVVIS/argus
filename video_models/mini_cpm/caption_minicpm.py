import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu


import argparse


def encode_video(video_path, num_frames):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    assert num_frames <= 64, "num_frames should be less than 64"
    frame_idx = uniform_sample(frame_idx, num_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    print("num frames:", len(frames))
    return frames


def load_model(args):

    if args.model_name == "minicpm_v_2_6":
        pretrained = "openbmb/MiniCPM-V-2_6"
    elif args.model_name == "minicpm_o_2_6":
        pretrained = "openbmb/MiniCPM-o-2_6"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    model = AutoModel.from_pretrained(
        pretrained,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )  # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    return {"model": model, "tokenizer": tokenizer}


def generate_caption(args, model_dict, prompt="Describe the video in great detail."):

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    frames = encode_video(args.clip_path, args.num_frames)
    print(f"Number Frames: {len(frames)}")

    msgs = [
        {"role": "user", "content": frames + [prompt]},
    ]

    # Set decode params for video
    params = {"temperature": args.model_temp}
    params["use_image_id"] = False
    params["max_slice_nums"] = 1  # use 1 if cuda OOM and video resolution > 448*448

    output = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)
    return output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="minicpm_v_2_6",
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
        for model_name in ["minicpm_v_2_6", "minicpm_o_2_6"]:
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
