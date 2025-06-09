import torch
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText
import cv2


def load_model(args):

    if args.model_name == "smolvlm2_2b":
        pretrained = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    elif args.model_name == "smolvlm2_500m":
        pretrained = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    elif args.model_name == "smolvlm2_256m":
        pretrained = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    processor = AutoProcessor.from_pretrained(pretrained)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2"
    ).to("cuda")

    return {"model": model, "processor": processor}


def get_desired_fps_given_video_path_and_num_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps * num_frames / num_frames_in_video


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    model, processor = model_dict["model"], model_dict["processor"]

    desired_fps = get_desired_fps_given_video_path_and_num_frames(
        args.clip_path, args.num_frames
    )

    print(f"Desired FPS: {desired_fps}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "path": args.clip_path,
                    "max_frames": args.num_frames,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        max_frames=args.num_frames,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(
        **inputs, do_sample=False, max_new_tokens=3000, temperature=args.model_temp
    )
    prompt_length = inputs["input_ids"].shape[1]

    user_prompt = processor.decode(
        inputs["input_ids"][0][:prompt_length], skip_special_tokens=True
    )
    print(f'User Prompt:\n"{user_prompt}"')

    answer = (
        processor.decode(
            generated_ids[0][prompt_length:],
            skip_special_tokens=True,
        )
        .strip("Assistant:")
        .strip()
    )

    return answer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="smolvlm2_2b",
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
        for model_name in ["smolvlm2_2b", "smolvlm2_500m", "smolvlm2_256m"]:
            args.model_name = model_name
            model_dict = load_model(args)
            caption = generate_caption(args, model_dict)
            print(f"Model: {model_name}")
            print(f"Caption: {caption}")
            print("-" * 50)
        return
    else:
        model_dict = load_model(args)
        caption = generate_caption(args, model_dict)
        print(caption)


if __name__ == "__main__":
    main()
