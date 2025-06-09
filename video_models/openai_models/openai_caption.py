import openai
import PIL.Image
import os
from openai import OpenAI
import os
import cv2
import base64
import argparse
import numpy as np
import io
from PIL import Image


def resize_image(img, target_size):
    # Open the image
    # Get original dimensions
    original_width, original_height = img.size

    # Calculate the ratio of the target size to the original size
    target_width, target_height = target_size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Use the smaller ratio to ensure the image fits within target dimensions
    ratio = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    print(
        f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}"
    )
    return resized_img


def extract_frames_with_timestamp(video_path, save_dir):
    FRAME_PREFIX = "frame"
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)  # Total duration in seconds
    # Extract and save frames
    for t in range(0, duration + 1):
        frame_number = int(round(t * fps))
        if frame_number >= total_frames:
            break
        # Set the video position to the frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()
        if not success:
            print(f"Warning: Could not read frame at {t} seconds.")
            continue
        # Format timestamp as MM:SS
        minutes = t // 60
        seconds = t % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        # Save the frame as a PNG image with timestamp in the filename
        filename = os.path.join(save_dir, f"{FRAME_PREFIX}{timestamp}.png")
        cv2.imwrite(filename, frame)
    # Release the video capture object
    video.release()
    print(f"Extracted frames to '{save_dir}'.")


def encode_image(image_path):
    if os.path.getsize(image_path) > 1e6:
        print(f"Got a large image: {image_path}")

        # Open the image and resize it
        image = PIL.Image.open(image_path)
        image = resize_image(image, (512, 512))
        image.save("tmp_img.png")
        image_path = f"tmp_img.png"

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_model(args):
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_KEY", "<your OpenAI API key if not set as env var>"
        )
    )
    return {"client": client}


def generate_caption(args, model_dict, prompt="Describe the video in great detail"):

    frames_save_dir = args.clip_name

    if os.path.isdir(frames_save_dir):
        print(f"Directory '{frames_save_dir}' already exists.")
    else:
        extract_frames_with_timestamp(args.clip_path, frames_save_dir)

    frames_list = sorted(os.listdir(frames_save_dir))

    base64_images = []
    frames_list = sorted(os.listdir(frames_save_dir))

    for filename in frames_list:
        if filename.endswith(".png"):
            image_path = os.path.join(frames_save_dir, filename)
            image_base64 = encode_image(image_path)
            base64_images.append(image_base64)

    ## if len(base64_images) > args.num_frames; uniformaly sample args.num_frames
    if len(base64_images) > args.num_frames:
        indices = np.linspace(0, len(base64_images) - 1, args.num_frames, dtype=int)
        base64_images = [base64_images[i] for i in indices]
        print(
            f"Uniformly sampled {args.num_frames} frames from {len(base64_images)} frames. [{indices}]"
        )

    ## make the content_dict
    content_dict = []
    for base64_image in base64_images:
        content_dict.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this video in detail.",
                },
                *content_dict,
            ],
        }
    ]

    client = model_dict["client"]
    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
    )

    caption = response.choices[0].message.content
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
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
        model_names = ["gpt-4o-mini", "gpt-4o"]
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
