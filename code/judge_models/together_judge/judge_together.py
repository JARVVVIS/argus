import os
import openai

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


def together_completion(client, model_id, conversation_log, temperature=0):

    response = client.chat.completions.create(
        model=model_id, messages=conversation_log, temperature=temperature
    )

    conversation_log.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content.strip(),
        }
    )
    return conversation_log


def judge_setup(args):
    ## save name to actual path
    if args.judge_model == "deepseek_r1":
        together_model = "deepseek-ai/DeepSeek-R1"
    elif args.judge_model == "deepseek_v3":
        together_model = "deepseek-ai/DeepSeek-V3"
    elif args.judge_model == "qwen_qwq":
        together_model = "Qwen/QwQ-32B-Preview"
    elif args.judge_model == "qwen_2_5_72b_instruct":
        together_model = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    elif args.judge_model == "llama_3_3_70b":
        together_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    else:
        raise NotImplementedError(f"Judge {args.judge_model} not implemented")

    print(f"Setup for {args.judge_model} with model {together_model}")

    together_temp = args.judge_temp
    client = openai.OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )
    return {
        "together_model": together_model,
        "together_temp": together_temp,
        "client": client,
    }


def judge_response(prompt, judge_setup):
    together_model = judge_setup["together_model"]
    together_temp = judge_setup["together_temp"]
    client = judge_setup["client"]

    prompt_conv = {"role": "user", "content": prompt}
    conversations = [prompt_conv]

    conversations = together_completion(
        client=client,
        model_id=together_model,
        conversation_log=conversations,
        temperature=together_temp,
    )

    response = conversations[-1]["content"].strip()

    return response
