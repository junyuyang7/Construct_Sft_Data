import logging
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments as TA

import torch

logger = logging.getLogger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)

fsdp_config = {
    "mpt7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "opt125m_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["OPTDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
    },
    "mpt7b_lora": {
        "fsdp_transformer_layer_cls_to_wrap": ["MPTBlock"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "llama2_13b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
    "mistral_7b_finetune": {
        "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "limit_all_gathers": "true",
        "use_orig_params": "true",
    },
}


@dataclass
class TrainingArguments(TA):
    analysis_mode: float = field(
        default=False,
        metadata={
            "help": (
                "Whether to run in analysis mode. "
            )
        },
    )
    analysis_dataset: str = field(
        default="bbh",
        metadata={
            "help": (
                "The dataset to use for analysis mode. "
            )
        },
    )
    train_dataset_names: str = field(
        default=None,
        metadata={
            "help": (
                "The dataset to use for training. "
            )
        },
    )

    def __post_init__(self):
        if isinstance(self.fsdp_config, str):
            self.fsdp_config = fsdp_config[self.fsdp_config]
        if self.train_dataset_names is not None:
            self.train_dataset_names = self.train_dataset_names.split(" ")
        super().__post_init__()

@dataclass
class DataArguments:
    train_files: List[str] = field(default_factory=list, metadata={
                                   "help": "The input training data files (multiple files in glob format)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    sample_data_seed: int = field(
        default=42, metadata={"help": ("The seed used for data sampling.")},
    )
    train_percentage: float = field(
        default=0.05, metadata={"help": ("Sampling percentage for each dataset")},
    )
    valid_percentage: float = field(
        default=1.0, metadata={"help": ("Sampling percentage for each dataset")},
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    ### added ####
    lora: Optional[bool] = field(default=False, metadata={
                                 "help": "whether to use lora"})
    lora_r: Optional[int] = field(default=8, metadata={"help": ("r for lora")})
    lora_alpha: Optional[float]=field(default=32, metadata={"help": ("alpha for lora")})
    lora_dropout: Optional[float]=field(default=0.1, metadata={"help": ("dropout for lora")})
    lora_target_modules: List[str]=field(
        default_factory=list, metadata={"help": ("target modules for lora")})

@dataclass
class GetGradArgument:
    """
    Arguments for obtaining validation gradients and related tasks.
    """
    mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "[train, valid], 计算 train or valid 的梯度"
        },
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the training data file we'd like to obtain the gradients/representations for. One of 'task' and 'train_file' must be specified."
        },
    )

    info_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "The type of information: choose from 'grads', 'reps', or 'loss'."
        },
    )

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the model."},
    )

    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of samples."},
    )

    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "The torch data type: choose from 'float32' or 'bfloat16'."
        },
    )

    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the output."},
    )

    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the data."},
    )

    gradient_projection_dimension: List[int] = field(
        default_factory=lambda: [8192],
        metadata={
            "help": "The dimension of the projection, can be a list of integers."
        },
    )

    gradient_type: str = field(
        default="adam",
        metadata={
            "help": "The type of gradient: choose from 'adam', 'sign', or 'sgd'."
        },
    )

    chat_format: str = field(
        default="tulu",
        metadata={"help": "The chat format."},
    )

    use_chat_format: bool = field(
        default=True,
        metadata={"help": "Whether to use chat format."},
    )

    max_length: int = field(
        default=2048,
        metadata={"help": "The maximum length."},
    )

    zh: bool = field(
        default=False,
        metadata={
            "help": "Whether to load the translated Chinese version of tydiqa dev data (Only applicable to tydiqa)."
        },
    )

    initialize_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the base model with LoRA; only works when is_peft is False."
        },
    )

    lora_r: int = field(
        default=8,
        metadata={"help": "The value of the LoRA `r` hyperparameter."},
    )

    lora_alpha: float = field(
        default=32,
        metadata={"help": "The value of the LoRA `alpha` hyperparameter."},
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The value of the LoRA `dropout` hyperparameter."},
    )

    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "The list of LoRA target modules."},
    )
