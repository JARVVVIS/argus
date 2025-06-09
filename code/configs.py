import importlib

## TODO: Set the ROOT_DIR to your local path
ROOT_DIR = "<YOUR_ROOT_DIR>/argus"

MODEL_MODULES = {
    "videochatgpt": "video_models.videochatgpt",
    "internvl2_1b": "video_models.internvl",
    "internvl2_2b": "video_models.internvl",
    "internvl2_4b": "video_models.internvl",
    "internvl2_8b": "video_models.internvl",
    "llava_video_7b": "video_models.llava_video",
    "llava_video_7b_video_only": "video_models.llava_video",
    "llava_next_video_7b_dpo": "video_models.llava_video",
    "llava_next_video_7b": "video_models.llava_video",
    "llava_ov_7b": "video_models.llava_ov",
    "llava_ov_0_5b_chat": "video_models.llava_ov",
    "minicpm_v_2_6": "video_models.mini_cpm",
    "mplug_owl3_7b": "video_models.mplug_owl",
    "mplug_owl3_1b": "video_models.mplug_owl",
    "qwen2_vl_2b": "video_models.qwen_vl",
    "qwen2_vl_2b_instruct": "video_models.qwen_vl",
    "smolvlm2_2b": "video_models.smolvlm2",
    "smolvlm2_500m": "video_models.smolvlm2",
    "smolvlm2_256m": "video_models.smolvlm2",
    "gemini-1.5-flash": "video_models.gemini",
    "gemini-2.0-pro-exp-02-05": "video_models.gemini",
    "gemini-2.0-flash": "video_models.gemini",
    "gpt-4o": "video_models.openai_models",
    "gpt-4o-mini": "video_models.openai_models",
    "qwen2_5_vl_7b_instruct": "video_models.qwen_vl",
    "qwen2_5_vl_3b_instruct": "video_models.qwen_vl",
}

JUDGE_MODULES = {
    "gpt-4o": "judge_models.openai_judge",
    "deepseek_r1": "judge_models.deepseek_judge",
    "deepseek_v3": "judge_models.deepseek_judge",
    "qwen_qwq": "judge_models.together_judge",
    "llama_3_3_70b": "judge_models.together_judge",
    "qwen_2_5_72b_instruct": "judge_models.together_judge",
}


def get_model_module(model_name):
    """Dynamically import only the needed model module based on model_name."""

    if model_name not in MODEL_MODULES.keys():
        raise NotImplementedError(f"Model {model_name} not implemented")

    return importlib.import_module(MODEL_MODULES[model_name])


def get_judge_module(judge_name):

    if judge_name not in JUDGE_MODULES.keys():
        raise NotImplementedError(f"Judge {judge_name} not implemented")

    return importlib.import_module(JUDGE_MODULES[judge_name])
