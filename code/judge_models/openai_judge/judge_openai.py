import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_KEY")

def openai_completion(client, model_id, conversation_log, temperature=0):

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
    openai_model = args.judge_model
    openai_temp = args.judge_temp
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return {
        "openai_model": openai_model,
        "openai_temp": openai_temp,
        "openai_client": openai_client,
    }


def judge_response(prompt, judge_setup):
    openai_model = judge_setup["openai_model"]
    openai_temp = judge_setup["openai_temp"]
    openai_client = judge_setup["openai_client"]

    prompt_conv = {"role": "user", "content": prompt}
    conversations = [prompt_conv]

    conversations = openai_completion(
        client=openai_client,
        model_id=openai_model,
        conversation_log=conversations,
        temperature=openai_temp,
    )

    response = conversations[-1]["content"].strip()

    return response
