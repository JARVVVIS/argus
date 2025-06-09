# ARGUS: Hallucination and Omission Evaluation for Video‑LLMs

ARGUS is a framework to calculate the degree of hallucination and omission in free-form video captions.

* **ArgusCost‑H** (or Hallucination-Cost) — degree of hallucinated content in the video-caption
* **ArgusCost‑O** (or Omission-Cost) — degree of omitted content in the video-caption

Lower values indicate better "performance".

---

## Repository Layout

```text
assets/                  # store the clips, prompts, etc
code/                    # implementation
    gen_model_caption.py
    nli_eval.py
    compute_metrics.py
    single_caption_inference.py
scripts/                 # exact commands to run
requirements.txt         # default environment
qwen_requirements.txt    # alt env for Qwen / SmolVLM
```

## Installation

```bash
git clone https://github.com/JARVVVIS/argus.git
cd argus

conda create -n argus python=3.11 -y
conda activate argus

## install torch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
# (if neeeded) install cudatoolkit
conda install -c conda-forge cudatoolkit-dev
## install flash-attn
pip install flash-attn --no-build-isolation

## rest of the dependencies
pip install -r requirements.txt
pip install -e .  # installs in‑house video_models for easy evaluations

# note: dedicated env. for Qwen / SmolVLM
# create a new env. and run previous commands except using "qwen_requirements.txt" instead of "requirements.txt"
# pip install -r qwen_requirements.txt
```

## Preparing the Data

To download the evaluation clips and place them under `assets/clips/`:

```bash
python code/download_argus_videos.py
```

## Basic Usage

First set `ROOT_DIR` in `code/configs.py`

Broad Steps: Generate captions → judge with NLI → compute metrics:

```bash
python code/gen_model_caption.py --model_name $model --clip_name $clip_name

python code/nli_eval.py --model_name $model --clip_name $clip_name

python code/compute_metrics.py --model_name $model --clip_name $clip_name
```

NOTE: If "clip_name" is not specified, it will use all clips in `assets/clips/`. Similarly, if "model_name" is not specified, it will use all models in `code/configs.py`.

Single‑caption inference:

```bash
python code/single_caption_inference.py \
  --caption "A panda eats bamboo." \
  --clip_path assets/clips/clip_12345.mp4
```

Ready‑made pipelines are available under `scripts/`.

## Citation

TODO: