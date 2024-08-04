import os
import os.path as osp
import sys
import argparse

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./mmsoc'))  

def parse_args(dataset_name: str = None, task: str = None, topic: str = None, text_input_length: int = None,
               model_name: str = None, split: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    if dataset_name is None:
        parser.add_argument("--dataset_name", type=str, choices=["memotion", "politi", "gossip", "YouTube_2000",
                                                                 "hatefulmemes"], default=None)

    if task is None:
        parser.add_argument("--task", type=str, choices=["humor", "sarcasm", "sentiment", "motivational",
                                                         "offensive",
                                                         "tag", "OCR",
                                                         "misinformation", "hatespeech", "Relevance"],
                            default=None)

    if model_name is None:
        parser.add_argument("--model_name", type=str,  # default="facebook/opt-350m"
                            default=None
                            )

    if topic is None:
        parser.add_argument("--topic", type=str, default=None)

    parser.add_argument("--load_in_4bits", action="store_true", help="Enable 4-bit inference")
    parser.add_argument("--load_in_8bits", action="store_true", help="Enable 8-bit inference")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    if split is None:
        parser.add_argument("--split", type=str, choices=["train", "val", "dev", "test"], default='test')

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--fill_null_values", action="store_true")
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--load_from_cache", action="store_true", help="Whether to load models and tokenizers from "
                                                                       "local "
                                                                       "cache. Useful when HuggingFace is inaccessible.")

    parser.add_argument("--num_rounds", type=int, default=2)

    # Available: "question", "explain", "caption"
    parser.add_argument("--question_types", nargs='+', default=["question", "explain"], help="")
    parser.add_argument("--idx_auth", type=int, default=0, help="Index of the authentication token to use")
    parser.add_argument("--result_file_type", type=str, choices=['xlsx', 'jsonl'], default='jsonl', help="Index of the "
                                                                                                         "authentication "
                                                                                                         "token to "
                                                                                                         "use")

    if text_input_length is None:
        parser.add_argument("--text_input_length", type=int, default=0, help="")

    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)

    if text_input_length is not None:
        args.text_input_length = text_input_length

    if dataset_name is not None:
        args.dataset_name = dataset_name

    if task is not None:
        args.task = task

    if topic is not None:
        args.topic = topic

    if model_name is not None:
        args.model_name = model_name

    if split is not None:
        args.split = split

    if args.model_name is None:
        model_name_without_orgs = None

    else:
        model_name_without_orgs = args.model_name.split('/')[-1]




    if args.dataset_name in ["memotion"]:
        if args.verbose:
            assert args.task in ["humor", "sentiment", "sarcasm", "OCR", "offensive"]

        print("Setting split to dev")

    elif args.dataset_name in ["YouTube"]:
        if args.verbose:
            assert args.task in ["topic", "tag"]

    elif args.dataset_name in ["hatefulmemes"]:
        if args.verbose:
            assert args.task in ["hatespeech", "OCR"]

        assert args.split != "test"

    elif args.dataset_name in ["politi", "gossip"]:
        if args.verbose:
            assert args.task in ["misinformation"] or args.topic in ["Social_Context", "Description"]

        print("Setting split to dev")

    if "caption" in args.question_types:
        assert "caption" in args.answers_file

    if args.load_in_4bits:
        args.load_in_8bits = False

    prefix = "captions_" if () else ""

    suffix = f"_{args.split}"

    if args.dataset_name is None:
        pass

    elif args.dataset_name in ["politi", "gossip"] or args.dataset_name.startswith("YouTube"):
        assert args.text_input_length > 0

    else:
        assert args.text_input_length == 0

    if args.text_input_length > 0:
        suffix += f"_UseTextInput{args.text_input_length}"

    if args.num_rounds >= 3:
        suffix += f"_num_rounds{args.num_rounds}"

    if model_name_without_orgs is not None:

        os.makedirs(f"{args.output_dir}/captions/{model_name_without_orgs}", exist_ok=True)
        os.makedirs(f"{args.output_dir}/{model_name_without_orgs}", exist_ok=True)
    if "caption" in args.question_types and len(args.question_types) == 1:
        args.answers_file = osp.join(args.output_dir, "captions", f"{model_name_without_orgs}",
                                     f"{prefix}{args.model_name.split('/')[-1]}_{args.dataset_name}_{args.task}{suffix}.xlsx")

    else:
        args.answers_file = None


    if args.result_file_type == "jsonl":
        # Finetuned LLaVA
        if args.task is not None:
            args.answers_file = osp.join(args.output_dir, f"{args.model_name.split('/')[-1]}",
                                         f"TASK_{model_name_without_orgs}_{args.dataset_name}_{args.task}{suffix}.jsonl")
        elif args.topic is not None:
            args.answers_file = osp.join(args.output_dir, f"{args.model_name.split('/')[-1]}",
                                         f"TOPIC_{model_name_without_orgs}_{args.dataset_name}_{args.topic}{suffix}.jsonl")

    elif args.result_file_type == "xlsx":
        args.answers_file = osp.join(args.output_dir, f"{model_name_without_orgs}",
                                     f"{model_name_without_orgs}_{args.dataset_name}_{args.task}{suffix}.xlsx")

    if args.model_name in ["GPT4V"]:
        args.answers_file = args.answers_file.replace('.jsonl', '.xlsx')


    return args
