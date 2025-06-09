import os
import openai

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")

def deepseek_completion(client, model_id, conversation_log, temperature=0):

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
        deepseek_model = "deepseek-reasoner"
    elif args.judge_model == "deepseek_v3":
        deepseek_model = "deepseek-chat"
    else:
        raise NotImplementedError(f"Judge {args.judge_model} not implemented")

    print(f"[Deepseek-Judge] Setup for {args.judge_model} with model {deepseek_model}")

    deepseek_temp = args.judge_temp
    client = openai.OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
    )

    return {
        "deepseek_model": deepseek_model,
        "deepseek_temp": deepseek_temp,
        "client": client,
    }


def judge_response(prompt, judge_setup):
    together_model = judge_setup["deepseek_model"]
    together_temp = judge_setup["deepseek_temp"]
    client = judge_setup["client"]

    prompt_conv = {"role": "user", "content": prompt}
    conversations = [prompt_conv]

    conversations = deepseek_completion(
        client=client,
        model_id=together_model,
        conversation_log=conversations,
        temperature=together_temp,
    )

    response = conversations[-1]["content"].strip()

    return response
