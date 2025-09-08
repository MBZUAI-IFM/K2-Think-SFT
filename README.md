# K2-Think SFT Recipe

This repository is initialized from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), adapted for supervised fine-tuning (SFT) in K2-Think recipe by [MBZUAI IFM](https://ifm.mbzuai.ac.ae/). The following sections document the dataset preparation and training configurations we used in this repository.

> **Note:** All paths in this documentation are relative to the repository root.

## Dataset

We used the Hugging Face dataset `a-m-team/AM-Thinking-v1-Distilled`, commit hash:
`3697c1829816a2b8d4d25995ed6d5d27ffb49b30` ([link](https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled/tree/3697c1829816a2b8d4d25995ed6d5d27ffb49b30))

### Preprocessing
Since the six subsets of this dataset do not share the same schema, directly calling `datasets.load_dataset()` raises errors. We provide a preprocessing script that aligns the format across subsets:

```bash
python get_am_dataset.py \
    --revision 3697c1829816a2b8d4d25995ed6d5d27ffb49b30 \
    data/AM-Thinking-3697c18.parquet
```

The processed dataset is saved as `data/AM-Thinking-3697c18.parquet` and registered in `data/dataset_info.json` under the name `"AM-Thinking-3697c18"`.

> **Important:** Run the preprocessing script first to generate the dataset file before training.

To enable loading from Parquet, we added a few lines to `src/llamafactory/data/loader.py`, marked with the comment `# Modified by K2-Think Team`.

## Training

### Chat Template
We registered the chat template under the name `"AM-thinking"` in `src/llamafactory/data/template.py`.

#### Special Tokens
- `<|im_start|>` - Start of message
- `<|im_end|>` - End of message

#### System Prompt
```
You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
```

#### Example Tokenized SFT Sample
```
<|im_start|>system
default system prompt<|im_end|>
<|im_start|>user
user content<|im_end|>
<|im_start|>assistant
assistant content<|im_end|>
```

### Configuration

#### SFT Config File
**File:** `examples/train_full/Qwen2.5-32B-base-AM-Thinking-v1-Distilled-3697c18.yaml`

#### Training Parameters
| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-32B |
| Sequence packing | True |
| Max sequence length | 32,768 |
| Batch size per GPU | 2 |
| Gradient accumulation | 1 |
| Learning rate | 1e-4 |
| Scheduler | cosine |
| Warmup ratio | 0.05 |
| Epochs | 2 |

#### Setup
> **Important:** Update these hardcoded paths in the config file for your environment:
> - `model_name_or_path`: Path to Qwen2.5-32B model
> - `deepspeed`: Path to DeepSpeed config file
> - `tokenized_path`: Path to tokenized dataset cache (created automatically on first training run)
> - `output_dir`: Path to save model checkpoints

### Distributed Training

#### Slurm Launch Script
**File:** `scripts/train_qwen_32_amthink_3697c18.sh`

#### Cluster Configuration
| Parameter | Value |
|-----------|-------|
| Nodes | 32 |
| GPUs per node | 8 |
| Global batch size | 512 (32 × 8 × 2) |

#### Setup
> **Important:** Update these paths in the script for your environment:
> - Conda environment path (please follow the instruction from [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to create the conda environment)
> - Hugging Face and Triton cache paths
> - LLaMA-Factory installation path
> - WandB API key
