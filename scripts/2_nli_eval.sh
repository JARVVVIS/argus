MODELS=(
    "minicpm_v_2_6"
    # "internvl2_1b"
    # "internvl2_2b"
    # "internvl2_4b"
    # "internvl2_8b"
    # "llava_video_7b" 
    # "llava_video_7b_video_only"
    # "llava_next_video_7b_dpo"
    # "llava_next_video_7b"
    # "mplug_owl3_7b" 
    # "mplug_owl3_1b"
    # "llava_ov_7b"
    # "llava_ov_0_5b_chat"
    # "gemini-1.5-pro"
    # "gemini-1.5-flash"
    # "gemini-2.0-flash-thinking-exp-01-21"
    # "gemini-2.0-flash-lite"
    # "gemini-2.0-pro-exp-02-05"
    # "gemini-2.0-flash"
    # "gpt-4o"
    # "gpt-4o-mini"
    )

for model in "${MODELS[@]}"; do
    python code/nli_eval.py \
        --model_name $model --clip_name "trimmed_chameleon"
done