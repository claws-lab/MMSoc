import argparse
import os
import os.path as osp
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('./mmsoc'))


def parse_args(dataset_name: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")

    if dataset_name is None:
        parser.add_argument("--dataset_name", type=str, choices=["memotion", "politi", "gossip", "YouTube_2000",
                                                                 "hatefulmemes"], default=None)

    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--answers_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--fill_null_values", action="store_true")
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--include_text_in_prompt", action="store_true",
                        help="Whether to include additional text in the prompt")

    parser.add_argument("--load_in_4bits", action="store_true", help="Enable 4-bit inference")
    parser.add_argument("--load_in_8bits", action="store_true", help="Enable 8-bit inference")

    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--load_from_cache", action="store_true", help="Whether to load models and tokenizers from "
                                                                       "local "
                                                                       "cache. Useful when HuggingFace is inaccessible.")

    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=2)

    # Available: "question", "explain", "caption"
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--question_types", nargs='+', default=["question", "explain"], help="")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--result_file_type", type=str, choices=['xlsx', 'jsonl'], default='jsonl', help="Index of the "
                                                                                                         "authentication "
                                                                                                         "token to "
                                                                                                         "use")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--split", type=str, choices=["train", "val", "dev", "test"], default='test')
    parser.add_argument("--task", type=str, choices=["humor", "sarcasm", "sentiment", "motivational",
                                                     "offensive",
                                                     "tag", "OCR",
                                                     "misinformation", "hatespeech", "Relevance"], default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--text_input_length", type=int, default=128, help="The number of tokens to consider in the "
                                                                           "output.")
    parser.add_argument("--verbose", action="store_true")


    args = parser.parse_args()

    args.data_dir = os.path.expanduser(args.data_dir)

    if dataset_name is not None:
        args.dataset_name = dataset_name

    args.model_name = args.model_path.split('/')[-1]

    if args.dataset_name in ["memotion"]:
        if args.verbose:
            assert args.task in ["humor", "sentiment", "sarcasm", "OCR", "offensive"]

        print("Setting split to dev")

    elif args.dataset_name in ["YouTube"]:
        if args.verbose:
            assert args.task in ["topic", "tag"]

    elif args.dataset_name in ["hatefulmemes"]:

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

    if args.text_input_length > 0:
        suffix += f"_UseTextInput{args.text_input_length}"

    if args.num_rounds >= 3:
        suffix += f"_num_rounds{args.num_rounds}"

    os.makedirs(f"{args.output_dir}/captions/{args.model_name}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.model_name}", exist_ok=True)

    if args.answers_file is None:
        args.answers_file = os.path.join(args.output_dir, args.dataset_name,
                                         f"answers_{args.model_name}_{args.task}.jsonl")

    os.makedirs(osp.dirname(args.answers_file), exist_ok=True)

    return args
