import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Stream-load and process AM-Thinking dataset, optionally at a specific revision."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Git revision (commit SHA or branch/tag) to load. "
            "If omitted, the default (latest) version is used."
        ),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the output Parquet file will be written."
    )
    args = parser.parse_args()

    # Build load_dataset kwargs
    load_kwargs = {
        "path": "a-m-team/AM-Thinking-v1-Distilled",
        "split": "train",
        "streaming": True,
    }
    if args.revision:
        load_kwargs["revision"] = args.revision

    # Load the dataset in streaming mode
    data = load_dataset(**load_kwargs)

    def process(example):
        return {
            "conversations": [
                {"from": turn["from"], "value": turn["value"]}
                for turn in example["conversations"]
            ]
        }

    # Materialize and process
    processed = []
    for example in tqdm(data, desc="Processing stream"):
        new_example = process(example)
        if new_example != None:
            processed.append(new_example)

    # Convert to Dataset and write Parquet
    processed_dataset = Dataset.from_list(processed)
    processed_dataset.to_parquet(args.output_path)
    print(f"Written {len(processed_dataset)} records to {args.output_path}")

if __name__ == "__main__":
    main()
