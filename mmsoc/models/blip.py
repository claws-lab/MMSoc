import json
import os
import sys
import traceback

import pandas as pd
from datasets import load_dataset
from tqdm import trange
from transformers import AutoProcessor, Blip2ForConditionalGeneration, InstructBlipForConditionalGeneration

sys.path.append(os.path.abspath('mmsoc'))

import const
from arguments import parse_args

from utils.misc_utils import project_setup, print_colored


def get_prompt():
    # TODO
    return ""


def save_results(answer_li, file_path):
    with pd.ExcelWriter(file_path) as writer:
        results_df = pd.DataFrame(answer_li)
        results_df.to_excel(writer, index=False)


def process_images_and_questions(data, args, processor, model):
    template = "Question: {} Answer: {}"
    batch_size = args.batch_size

    ans_file = open(args.answers_file, "w")

    for start_index in trange(0, len(data), batch_size):
        num_examples_in_batch = min(batch_size, len(data) - start_index)

        try:
            entries = data[start_index:start_index + num_examples_in_batch]
            batch_images = entries['image']
            # texts = [truncate_input(text, args.max_seq_length) for text in entries['text']]
            input_ids = []
            batch_questions = [[] for _ in range(num_examples_in_batch)]
            batch_outputs = [[] for _ in range(num_examples_in_batch)]
            batch_context = [[] for _ in range(num_examples_in_batch)]
            batch_inputs = []
            batch_texts = []

            batch_inputs = []

            for example_index in range(num_examples_in_batch):
                # image = entries['image']
                # text = get_prompt()
                question = f"{const.TASK2QUESTION[args.task]}"
                if args.include_text_in_prompt:
                    text = f"{text} {entries['text']}"

                # context = batch_context[example_index]

                batch_questions[example_index] += [question]

                prompt = "Question: " + question + " Answer:"

                # # context = batch_context[example_index]
                # prompt = " ".join([template.format(context[i][0], context[i][1]) for i in
                #                     range(len(context))]) + " Question: " + question + " Answer:"
                prompt = prompt.strip(" ")
                batch_inputs += [prompt]

            batch_input_ids = processor(batch_images,
                                        text=batch_inputs,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **batch_input_ids,
                num_beams=5,
                max_new_tokens=args.max_seq_length,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )

            generated_ids[generated_ids == -1] = processor.tokenizer.pad_token_id

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for example_index in range(num_examples_in_batch):
                outputs = generated_text[example_index]
                outputs = outputs[0].strip()

                print_colored(f"Prompt {start_index + example_index}: {batch_inputs[start_index + example_index]}",
                              "blue")
                print_colored(outputs, "yellow")

                batch_context[example_index] += [(batch_questions[example_index][0], outputs)]
                batch_outputs[example_index] += [outputs]

                ans_file.write(json.dumps({"question_id": start_index + example_index,
                                           "prompt": batch_inputs[start_index + example_index],
                                           "text": outputs,
                                           "output": batch_outputs[example_index]}) + "\n")
                ans_file.flush()

        except Exception as e:
            traceback.print_exc()

    ans_file.close()


def main():
    project_setup()
    args = parse_args()
    data = load_dataset("Ahren09/Memotion", split='validation')

    if args.debug:
        processor, model = None, None


    elif "instructblip" in args.model_path:
        """
        Supported models:
            instructblip-vicuna-7b
            instructblip-vicuna-13b
            instructblip-flan-t5-xl
            instructblip-flan-t5-xxl
        """

        processor = AutoProcessor.from_pretrained(args.model_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            args.model_path, device_map=args.device, load_in_8bit=args.load_in_8bits, use_auth_token=True)

    elif "blip2" in args.model_path:
        """
        Supported models:
            Salesforce/blip2-flan-t5-xl
            Salesforce/blip2-flan-t5-xxl
            Salesforce/blip2-opt-2.7b
            Salesforce/blip2-opt-6.7b-coco
        
        """

        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_path, device_map=args.device, load_in_8bit=args.load_in_8bits, use_auth_token=True)

    else:
        raise ValueError(f"Invalid model path: {args.model_path}")

    # df, _ = load_data(args.dataset_name, args.split, args.data_dir)

    answers = process_images_and_questions(data, args, processor, model)
    save_results(answers, args.answers_file)


if __name__ == "__main__":
    main()
