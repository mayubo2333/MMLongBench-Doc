import os
import re
import math
import json
import argparse
import fitz
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from eval.extract_answer import extract_answer


def load_model(model_name, cache_path):
    if model_name == '4khd':
        from models.internlm_xc2_4khd import init_model, get_response_concat
    elif model_name == 'internvl':
        from models.internvl_chat import init_model, get_response_concat
    elif model_name == 'minicpm_llama3':
        from models.minicpm_llama3 import init_model, get_response_concat
    else:
        raise NotImplementedError
    model = init_model(cache_path)
    return model, get_response_concat


def extract_images(sample, document_path, max_pages=1000, resolution=144):
    image_list = list()
    doc_name = re.sub("\.pdf$", "", sample["doc_id"]).split("/")[-1]
    with fitz.open(os.path.join(document_path, sample["doc_id"])) as pdf:
        for index, page in enumerate(pdf[:max_pages]):
            if not os.path.exists(f"./tmp/{doc_name}_{index+1}.png"):
                im = page.get_pixmap(dpi=resolution)
                im.save(f"./tmp/{doc_name}_{index+1}.png")
            image_list.append(f"./tmp/{doc_name}_{index+1}.png")

    return image_list


def concat_images(image_list, concat_num=1, column_num=3):
    interval = max(math.ceil(len(image_list) / concat_num), 1)
    concatenated_image_list = list()

    for i in range(0, len(image_list), interval):
        image_path = "_".join(image_list[0].split("_")[:-1]) + "_concat{}_{}.jpg".format(concat_num, i//interval)
        if not os.path.exists(image_path):
            images_this_batch = [
                Image.open(filename) for filename in image_list[i:i + interval]
            ]
            if column_num==1:
                total_height = images_this_batch[0].height*len(images_this_batch)
            else:
                total_height = images_this_batch[0].height*((len(images_this_batch)-1)//column_num+1)

            concatenated_image = Image.new('RGB', (images_this_batch[0].width*column_num, total_height), 'white')
            x_offset, y_offset = 0, 0
            for cnt, image in enumerate(images_this_batch):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (cnt+1)%column_num==0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_image.save(image_path)
        concatenated_image_list.append(image_path)

    return concatenated_image_list


def load_questions(args):
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)
    # load evaluation prompt
    with open("./eval/prompt_for_answer_extraction.md") as f:
        prompt = f.read()

    model, get_response_concat = load_model(args.model_name, args.model_cached_path)

    for sample in tqdm(samples):
        if "score" in sample:
            score = sample["score"]
        else:
            image_list = extract_images(sample, document_path=args.document_path, max_pages=args.max_pages, resolution=args.resolution)
            concat_image_list = concat_images(image_list, concat_num=args.concat_num)
            response = get_response_concat(model, sample["question"], concat_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)

            if response == 'Failed':
                tmp_concat_num = args.concat_num - 1
                while response == 'Failed' and tmp_concat_num > 0:
                    concat_image_list = concat_images(image_list, concat_num=tmp_concat_num)
                    response = get_response_concat(model, sample["question"], concat_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)
                    tmp_concat_num -= 1

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

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))
        
        with open(args.output_path, 'w') as f:
            json.dump(samples, f)
    
    show_results(samples, show_path=re.sub("\.json$", ".txt", args.output_path))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/samples.json")
    parser.add_argument("--document_path", type=str, default="./data/documents")
    parser.add_argument("--extractor_prompt_path", type=str, default="./eval/prompt_for_answer_extraction.md")
    parser.add_argument("--model_name", type=str, default="internvl", choices=["internvl", "4khd", "minicpm_llama3"])
    parser.add_argument("--model_cached_path", type=str, default=None)
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    
    args.output_path = f'./results/res_{args.model_name}.json'
    args.concat_num = 1 if args.model_name in ['minicpm_llama3'] else 5
    load_questions(args)