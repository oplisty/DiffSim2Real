import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import torch
from openai import OpenAI
from transformers import AutoProcessor, AutoModelForVision2Seq

from qwen_vl_utils import process_vision_info
from prompt.prompt import Prompt

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

DEFAULT_LOCAL_MODEL_PATH=""

def decompose_video(video_path: str = "", save_path: Optional[str] = None) -> str:
    """
    Decompose a video a series of frame if save_path is None save in 
    
    video
    |-video.mp4
    |-video_frame
        |---frame001.png
        ...
    """
    if not video_path:
        raise ValueError("`video_path` cannot be empty.")

    video_path = Path(video_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if save_path is None:
        save_path = str(video_path.with_suffix("")) + "_frames"

    save_path = Path(save_path).resolve()
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_file = save_path / f"frame{frame_idx:06d}.png"
        cv2.imwrite(str(frame_file), frame)

    cap.release()
    return str(save_path)


def inference_with_openai_api(
    video: Dict,
    prompt: str,
) -> str:
    """
    Run video understanding inference through an OpenAI-compatible API.

    Args:
        video (Dict): Video payload. Expected keys:
            - frame_list: list of frame paths or URLs
            - fps: original sampling fps
        prompt (str): Prompt text.
        model_id (str): Model name served by the API.

    Returns:
        str: Model response text.
    """
    api_key = "sk-14feb0aba5814c62960c2dc4e1d17f4d"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    video_msg = {
        "type": "video",
        "video": video["frame_list"],
    }

    messages = [
        {
            "role": "user",
            "content": [
                video_msg,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    completion = client.chat.completions.create(
        model="qwen3-vl-32b-thinking",
        messages=messages,
    )

    return completion.choices[0].message.content


def save_info(
    inv_description: str,
    positive_description: str,
    video_path: str,
    output_path: Optional[str] = None,
) -> str:
    video_path_obj = Path(video_path).resolve()

    if output_path is None:
        json_path = video_path_obj.parent.parent / f"{video_path_obj.parent.name}.json"
    else:
        json_path = Path(output_path).resolve()

    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "video": {
                "video_path": str(video_path_obj),
                "file_name": video_path_obj.name,
            }
        }

    data["inverse"] = inv_description
    data["positive"] = positive_description

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(json_path)



def inference(
    video: Union[str, List],
    prompt: str,
    model_path: str,
    max_new_tokens: int = 2048,
    total_pixels: int = 20480 * 32 * 32,
    min_pixels: int = 64 * 32 * 32,
    max_frames: int = 2048,
    sample_fps: int = 2,
) -> str:
    """
    Perform multimodal inference on input video and text prompt.

    Args:
        video (str or list): Video path/URL or a pre-sampled frame list.
        prompt (str): Prompt text.
        model_path (str): Local model path.
        max_new_tokens (int): Maximum generation length.
        total_pixels (int): Maximum total pixels for resizing.
        min_pixels (int): Minimum total pixels for resizing.
        max_frames (int): Maximum frame count.
        sample_fps (int): Sampling FPS for pre-sampled frame lists.

    Returns:
        str: Generated response.
    """
    if not model_path:
        raise ValueError("`model_path` must be provided for local inference.")

    processor = AutoProcessor.from_pretrained(model_path)

    model, _ = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        output_loading_info=True,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video,
                    "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_frames": max_frames,
                    "sample_fps": sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_id[len(input_id):]
        for input_id, output_id in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return output_text[0]


def collect_frame_list(frame_path: str) -> List[str]:
    """
    Collect image frame files from a directory.
    """
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    frame_dir = Path(frame_path)

    if not frame_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_path}")

    frame_list = sorted(
        str(frame_dir / f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(exts)
    )

    if not frame_list:
        raise ValueError(f"No image frames found in: {frame_path}")
    return frame_list


def main(args):
    video_path = args.input_video_path
    output_json = args.output_json
    use_api = args.use_api

    frame_path = decompose_video(video_path)
    video_frame_list = collect_frame_list(frame_path)

    video = {
        "frame_list": video_frame_list,
        "fps": 0.5,
    }

    inverse_prompt = Prompt["inversion_prompt"]["prompt"]
    positive_prompt = Prompt["positive_prompt"]["prompt"]

    if use_api:
        video_content = inference_with_openai_api(video, inverse_prompt)
        detail_content = inference_with_openai_api(video, positive_prompt)
    else:
        if not DEFAULT_LOCAL_MODEL_PATH:
            raise ValueError(
                "Local inference is selected, but `LOCAL_VL_MODEL_PATH` is not set."
            )
        video_content = inference(video_frame_list, inverse_prompt, model_path=DEFAULT_LOCAL_MODEL_PATH, sample_fps=2)
        detail_content = inference(video_frame_list, positive_prompt, model_path=DEFAULT_LOCAL_MODEL_PATH, sample_fps=2)

    save_info(video_content,detail_content,video_path,output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video understanding pipeline")
    parser.add_argument(
        "--input_video_path",
        required=True,
        help="Path to the input HDF5 video file",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save merged JSON output",
    )
    parser.add_argument(
        "--use_api",
        action="store_true",
        help="Use OpenAI-compatible API for inference",
    )
    args = parser.parse_args()
    main(args)
