import os
import re
import json
import base64
import argparse
import fitz

from PIL import Image
from uuid import uuid4
from tqdm import tqdm

from eval.extract_answer import extract_answer
from eval.eval_score import eval_score


def encode_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret


def process_sample_gpt(sample, args):
    question = sample["question"]
    doc_name = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]

    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:args.max_pages]):
            if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                image = page.get_pixmap(dpi=args.resolution)
                image.save(f"./tmp/{doc_name}_{index+1}.png")
            image = Image.open(f"./tmp/{doc_name}_{index+1}.png")
            encoded_image = encode_image_to_base64(image)
            image_list.append(encoded_image)

    content = list()
    content.append(
        {
            "type": "text",
            "text": question,
        }
    )
    for encoded_image in image_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages


def process_sample_gemini(sample, args):
    question = sample["question"]
    doc_name = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]

    image_list = list()
    with fitz.open(os.path.join(args.document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:args.max_pages]):
            if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                im = page.get_pixmap(dpi=args.resolution)
                im.save(f"./tmp/{doc_name}_{index+1}.png")
            image_list.append(f"./tmp/{doc_name}_{index+1}.png")
    
    return [question] + image_list


def process_sample(sample, args):
    if "gpt-4" in args.model_name:
        return process_sample_gpt(sample, args)
    elif "gemini-1.5" in args.model_name:
        return process_sample_gemini(sample, args)
    else:
        raise AssertionError()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/samples.json")
    parser.add_argument("--document_path", type=str, default="./data/documents")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=360)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extractor_prompt_path", type=str, default="./eval/prompt_for_answer_extraction.md")
    args = parser.parse_args()

    args.output_path = f'./results/res_{args.model_name}.json'

    if "gpt-4" in args.model_name:
        from openai import OpenAI
        client = OpenAI()
    elif "gemini-1.5" in args.model_name:
        import google.generativeai as genai
        client = genai.GenerativeModel(args.model_name)
    else:
        raise AssertionError()

    with open(args.extractor_prompt_path) as f:
        prompt = f.read()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)

    all_score = 0.0
    for cnt, sample in enumerate(tqdm(samples)):
        if "score" in sample:
            score = sample["score"]
        else:
            messages = process_sample(sample, args)
            
            try_cnt = 0
            is_success = False
            while True:
                try:
                    if "gpt-4" in args.model_name:
                        response = client.chat.completions.create(
                            model=args.model_name,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature
                        )
                        response = response.choices[0].message.content
                    elif "gemini-1.5" in args.model_name:
                        response = client.generate_content(messages)
                        response.resolve()
                        response = response.text.strip()
                    else:
                        pass
                    is_success = True
                except:
                    try_cnt += 1
                    response = "Failed"
                if is_success or try_cnt>args.max_try:
                    break
                
            sample["response"] = response
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res
            try:
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            except:
                pred_ans = "Failed to extract"
                score = 0.0
            sample["pred"] = pred_ans
            sample["score"] = score

        all_score += score
        avg_score = all_score/(cnt+1)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg score: {}".format(avg_score))
        
        with open(args.output_path, 'w') as f:
            json.dump(samples, f)