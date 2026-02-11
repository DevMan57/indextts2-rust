import argparse
import json
import re
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPT = "文本情感分类"
CN_KEY_TO_EN = {
    "高兴": "happy",
    "愤怒": "angry",
    "悲伤": "sad",
    "恐惧": "afraid",
    "反感": "disgusted",
    "低落": "melancholic",
    "惊讶": "surprised",
    "自然": "calm",
}
ORDER = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
MELANCHOLIC_WORDS = {
    "低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"
}


def clamp_score(value, min_score=0.0, max_score=1.2):
    return max(min_score, min(max_score, value))


def convert(content):
    emotion_dict = {
        CN_KEY_TO_EN[cn_key]: clamp_score(content.get(cn_key, 0.0))
        for cn_key in ORDER
    }
    if all(val <= 0.0 for val in emotion_dict.values()):
        emotion_dict["calm"] = 1.0
    return emotion_dict


def parse_content(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            m.group(1): float(m.group(2))
            for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', text)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="float16",
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": args.text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
    content = parse_content(content)

    text_input_lower = args.text.lower()
    if any(word in text_input_lower for word in MELANCHOLIC_WORDS):
        content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)

    emotion_dict = convert(content)
    vec = [emotion_dict[k] for k in ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]]
    print(json.dumps(vec, ensure_ascii=False))


if __name__ == "__main__":
    main()
