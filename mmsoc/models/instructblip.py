import json
import os
import sys
import traceback
import pandas as pd
from PIL import Image
from tqdm import trange
from transformers import AutoProcessor, Blip2ForConditionalGeneration

sys.path.append(os.path.abspath('mmsoc'))

import const
from arguments import parse_args
from utils.data_utils import load_data, get_image_path

from utils.model_utils import truncate_input
from utils.misc_utils import project_setup, print_colored



def load_model(args, device):
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name, device_map=device, load_in_8bit=args.load_in_8bits, use_auth_token=True)
    return processor, model

def save_results(answer_li, file_path):
    with pd.ExcelWriter(file_path) as writer:
        results_df = pd.DataFrame(answer_li)
        results_df.to_excel(writer, index=False)

def process_images_and_questions(df, args, processor, model, device):
    template = "Question: {} Answer: {}."
    ans_list = []

    for idx in trange(0, len(df), args.batch_size):
        try:
            process_batch(df, idx, args, processor, model, device, template, ans_list)
        except Exception as e:
            traceback.print_exc()

    return ans_list

def process_batch(df, start_idx, args, processor, model, device, template, ans_list):
    num_examples_in_batch = min(args.batch_size, len(df) - start_idx)
    for j in range(num_examples_in_batch):
        process_single_example(df, start_idx + j, args, processor, model, device, template, ans_list)

def process_single_example(df, idx, args, processor, model, device, template, ans_list):
    line = df.iloc[idx]
    image_path = get_image_path(args, line)
    image = Image.open(image_path)
    text = truncate_input(line['text'], args.max_seq_length)
    prompt = template.format(text, "")
    input_ids = processor(image, text=prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, max_new_tokens=128)
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    ans_list.append({"question_id": idx, "prompt": prompt, "text": output_text})

def main():
    project_setup()
    args = parse_args()
    device = args.device
    processor, model = load_model(args, device)
    df, _ = load_data(args.dataset_name, args.split, args.data_dir)
    answers = process_images_and_questions(df, args, processor, model, device)
    save_results(answers, args.answers_file)

if __name__ == "__main__":
    main()
