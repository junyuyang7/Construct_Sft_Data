import os
from dataclasses import dataclass
import time

SENTENCEBERT_MODEL = 'paraphrase-MiniLM-L6-v2' # 计算 KNN-i 指标

device_ids = '0,1'
llm_model_dict = {
    'chatglm3-6b': "/home/yangjy/Study/ChatAgent_RAG/llm_models/chatglm3-6b/",
    'llama-2-7b-chat': "/home/yangjy/Study/ChatAgent_RAG/llm_models/Llama-2-7b-chat-hf/",
    "llama-2-13b-chat": "/home/yangjy/Study/ChatAgent_RAG/llm_models/Llama-2-13b-chat-hf/",
    "llama-3-8b-instruct": "/home/yangjy/Study/ChatAgent_RAG/llm_models/Meta-Llama-3.1-8B-Instruct/",
    "qwen2-7b": "/home/yangjy/Study/ChatAgent_RAG/llm_models/Qwen2-7B-Instruct/",
}

@dataclass
class Args:
    model_path: str = llm_model_dict['chatglm3-6b']
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    repeat: int = None
    total_prompts: int = 1000
    max_tokens: int = 1024
    max_model_len: int = 4096
    early_stopping: bool = True
    disable_early_stopping: bool = False
    system_prompt: bool = False
    sanitize: bool = False
    logits_processor: bool = False
    control_tasks: str = None
    shuffle: bool = True
    skip_special_tokens: bool = True
    checkpoint_every: int = 100
    engine: str = "vllm"
    device: str = "1"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.95
    swap_space: float = 2.0
    output_folder: str = "../test_data"
    job_name: str = None
    timestamp: int = int(time.time())
    seed: int = None  # Random seed
    max_memory: str = '10GiB'

stop_tokens_dict = {
  "chatglm3-6b": {
    "stop_tokens": [
      "[gMASK]",
      "sop",
      "<|user|>",
      "<|assistant|>"
    ],
    "stop_token_ids": [
      64790,
      64792,
      64795,
      64796
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "llama-3-8b-instruct": {
    "stop_tokens": [
      "<|eot_id|>",
      "<|end_of_text|>",
      "<|start_header_id|>",
      "<|end_header_id|>"
    ],
    "stop_token_ids": [
      128009,
      128001,
      128006,
      128007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ]
  },
  "meta-llama/Meta-Llama-3.1-8B-Instruct": {
    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "stop_tokens": [
      "<|eot_id|>",
      "<|end_of_text|>",
      "<|start_header_id|>",
      "<|end_header_id|>"
    ],
    "stop_token_ids": [
      128009,
      128001,
      128006,
      128007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "meta-llama/Meta-Llama-3-70B-Instruct": {
    "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",
    "stop_tokens": [
      "<|eot_id|>",
      "<|end_of_text|>",
      "<|start_header_id|>",
      "<|end_header_id|>"
    ],
    "stop_token_ids": [
      128009,
      128001,
      128006,
      128007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
    },
  "meta-llama/Meta-Llama-3.1-70B-Instruct": {
    "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "stop_tokens": [
      "<|eot_id|>",
      "<|end_of_text|>",
      "<|start_header_id|>",
      "<|end_header_id|>"
    ],
    "stop_token_ids": [
      128009,
      128001,
      128006,
      128007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "meta-llama/Meta-Llama-3.1-405B-Instruct": {
    "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "stop_tokens": [
      "<|eot_id|>",
      "<|end_of_text|>",
      "<|starter_header_id|>",
      "<|end_header_id|>"
    ],
    "stop_token_ids": [
      128009,
      128001,
      128006,
      128007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "lmsys/vicuna-7b-v1.5": {
    "model_name": "lmsys/vicuna-7b-v1.5",
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>"
    ],
    "stop_token_ids": [
      2,
      1,
      0
    ],
    "stop_tokens_assistant": [
      "ASSISTANT"
    ],
  },
  "llama-2-7b-chat": {
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>"
    ],
    "stop_token_ids": [
      2,
      1,
      0
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "llama-2-13b-chat": {
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>"
    ],
    "stop_token_ids": [
      2,
      1,
      0
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "NousResearch/Nous-Hermes-llama-2-7b": {
    "model_name": "NousResearch/Nous-Hermes-llama-2-7b",
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>"
    ],
    "stop_token_ids": [
      2,
      1,
      0
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
    "pre_query_template": "### Instruction:\n\n"
  },
  "qwen2-7b": {
    "stop_tokens": [
      "<|im_start|>",
      "<|im_end|>",
      "<|endoftext|>"
    ],
    "stop_token_ids": [
      151643,
      151644,
      151645
    ],
    "stop_tokens_assistant": [
      "Assistant",
      "assistant"
    ],
  },
  "Qwen/Qwen2-7B-Instruct": {
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "stop_tokens": [
      "<|im_start|>",
      "<|im_end|>",
      "<|endoftext|>"
    ],
    "stop_token_ids": [
      151643,
      151644,
      151645
    ],
    "stop_tokens_assistant": [
      "Assistant",
      "assistant"
    ],
  },
  "Qwen/Qwen2-Math-7B-Instruct": {
    "model_name": "Qwen/Qwen2-Math-7B-Instruct",
    "stop_tokens": [
      "<|im_start|>",
      "<|im_end|>",
      "<|endoftext|>"
    ],
    "stop_token_ids": [
      151643,
      151644,
      151645
    ],
    "stop_tokens_assistant": [
      "Assistant",
      "assistant"
    ],
  },
  "Qwen/Qwen1.5-7B-Chat": {
    "model_name": "Qwen/Qwen1.5-7B-Chat",
    "stop_tokens": [
      "<|im_start|>",
      "<|im_end|>",
      "<|endoftext|>"
    ],
    "stop_token_ids": [
      151643,
      151644,
      151645
    ],
    "stop_tokens_assistant": [
      "Assistant",
      "assistant"
    ],
  },
  "mistralai/Mistral-7B-Instruct-v0.3": {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
    "stop_tokens": [
      "<s>",
      "</s>",
      "[INST]",
      "[/INST]"
    ],
    "stop_token_ids": [
      1,
      2,
      3,
      4
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "google/gemma-1.1-7b-it": {
    "model_name": "google/gemma-1.1-7b-it",
    "stop_tokens": [
      "<eos>",
      "<bos>",
      "<start_of_turn>",
      "<end_of_turn>"
    ],
    "stop_token_ids": [
      1,
      2,
      106,
      107
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "google/gemma-2-27b-it": {
    "model_name": "google/gemma-2-27b-it",
    "stop_tokens": [
      "<eos>",
      "<bos>",
      "<end_of_turn>"
    ],
    "stop_token_ids": [
      1,
      2,
      107
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "google/gemma-2-9b-it": {
    "model_name": "google/gemma-2-9b-it",
    "stop_tokens": [
      "<eos>",
      "<bos>",
      "<end_of_turn>"
    ],
    "stop_token_ids": [
      1,
      2,
      107
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "microsoft/Phi-3-mini-128k-instruct": {
    "model_name": "microsoft/Phi-3-mini-128k-instruct",
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>",
      "<|endoftext|>",
      "<|user|>",
      "<|assistant|>",
      "<|system|>",
      "<|end|>"
    ],
    "stop_token_ids": [
      2,
      1,
      0,
      32000,
      32001,
      32006,
      32007,
      32010
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "microsoft/Phi-3-small-128k-instruct": {
    "model_name": "microsoft/Phi-3-small-128k-instruct",
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>",
      "<|endoftext|>",
      "<|user|>",
      "<|assistant|>",
      "<|system|>",
      "<|end|>"
    ],
    "stop_token_ids": [
      2,
      1,
      0,
      32000,
      32001,
      32006,
      32007,
      32010
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "microsoft/Phi-3-medium-128k-instruct": {
    "model_name": "microsoft/Phi-3-medium-128k-instruct",
    "stop_tokens": [
      "</s>",
      "<s>",
      "<unk>",
      "<|endoftext|>",
      "<|user|>",
      "<|assistant|>",
      "<|system|>",
      "<|end|>"
    ],
    "stop_token_ids": [
      2,
      1,
      0,
      32000,
      32001,
      32006,
      32007,
      32010
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "01-ai/Yi-1.5-34B-Chat": {
    "model_name": "01-ai/Yi-1.5-34B-Chat",
    "stop_tokens": [
      "<|startoftext|>",
      "<|endoftext|>",
      "<|im_end|>"
    ],
    "stop_token_ids": [
      1,
      2,
      7
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  },
  "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {
    "model_name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "stop_tokens": [
      "<｜begin▁of▁sentence｜>",
      "<｜end▁of▁sentence｜>",
      "<|EOT|>",
      "<｜User｜>",
      "<｜Assistant｜>"
    ],
    "stop_token_ids": [
      100000,
      100001,
      100008,
      100006,
      100007
    ],
    "stop_tokens_assistant": [
      "assistant"
    ],
  }
}

ONLINE_LLM_MODEL = {
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },

    # 智谱AI API,具体注册及api key获取请前往 http://open.bigmodel.cn
    "zhipu-api": {
        "api_key": "",
        "version": "glm-4",
        "provider": "ChatGLMWorker", # 即server model_worker中的类
    },

    # 具体注册及api key获取请前往 https://api.minimax.chat/
    "minimax-api": {
        "group_id": "",
        "api_key": "",
        "is_pro": False,
        "provider": "MiniMaxWorker",
    },

    # 具体注册及api key获取请前往 https://xinghuo.xfyun.cn/
    "xinghuo-api": {
        "APPID": "",
        "APISecret": "",
        "api_key": "",
        "version": "v3.5", # 你使用的讯飞星火大模型版本，可选包括 "v3.5","v3.0", "v2.0", "v1.5"
        "provider": "XingHuoWorker",
    },

    # 百度千帆 API，申请方式请参考 https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
    "qianfan-api": {
        "version": "ERNIE-Bot",  # 注意大小写。当前支持 "ERNIE-Bot" 或 "ERNIE-Bot-turbo"， 更多的见官方文档。
        "version_url": "",  # 也可以不填写version，直接填写在千帆申请模型发布的API地址
        "api_key": "",
        "secret_key": "",
        "provider": "QianFanWorker",
    },

    # 火山方舟 API，文档参考 https://www.volcengine.com/docs/82379
    "fangzhou-api": {
        "version": "chatglm-6b-model",
        "version_url": "",
        "api_key": "",
        "secret_key": "",
        "provider": "FangZhouWorker",
    },

    # 阿里云通义千问 API，文档参考 https://help.aliyun.com/zh/dashscope/developer-reference/api-details
    "qwen-api": {
        "version": "qwen-max",
        "api_key": "",
        "provider": "QwenWorker",
        "embed_model": "text-embedding-v1"  # embedding 模型名称
    },

    # 百川 API，申请方式请参考 https://www.baichuan-ai.com/home#api-enter
    "baichuan-api": {
        "version": "Baichuan2-53B",
        "api_key": "",
        "secret_key": "",
        "provider": "BaiChuanWorker",
    },

    # Azure API
    "azure-api": {
        "deployment_name": "",  # 部署容器的名字
        "resource_name": "",  # https://{resource_name}.openai.azure.com/openai/ 填写resource_name的部分，其他部分不要填写
        "api_version": "",  # API的版本，不是模型版本
        "api_key": "",
        "provider": "AzureWorker",
    },

    # 昆仑万维天工 API https://model-platform.tiangong.cn/
    "tiangong-api": {
        "version": "SkyChat-MegaVerse",
        "api_key": "",
        "secret_key": "",
        "provider": "TianGongWorker",
    },
    # Gemini API https://makersuite.google.com/app/apikey
    "gemini-api": {
        "api_key": "",
        "provider": "GeminiWorker",
    }

}

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型

MODEL_PATH = {
    "embed_model": {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
        "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
        "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
        "text2vec-bge-large-chinese": "shibing624/text2vec-bge-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",

        "bge-small-zh": "BAAI/bge-small-zh",
        "bge-base-zh": "BAAI/bge-base-zh",
        "bge-large-zh": "BAAI/bge-large-zh",
        "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
        "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
        "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",

        "bge-m3": "BAAI/bge-m3",

        "piccolo-base-zh": "sensenova/piccolo-base-zh",
        "piccolo-large-zh": "sensenova/piccolo-large-zh",
        "nlp_gte_sentence-embedding_chinese-large": "damo/nlp_gte_sentence-embedding_chinese-large",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },

    "llm_model": {
        "chatglm2-6b": "THUDM/chatglm2-6b",
        "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
        "chatglm3-6b": "THUDM/chatglm3-6b",
        "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

        "Orion-14B-Chat": "OrionStarAI/Orion-14B-Chat",
        "Orion-14B-Chat-Plugin": "OrionStarAI/Orion-14B-Chat-Plugin",
        "Orion-14B-LongChat": "OrionStarAI/Orion-14B-LongChat",

        "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

        "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
        "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
        "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
        "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",

        # Qwen1.5 模型 VLLM可能出现问题
        "Qwen1.5-0.5B-Chat": "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen1.5-1.8B-Chat": "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen1.5-4B-Chat": "Qwen/Qwen1.5-4B-Chat",
        "Qwen1.5-7B-Chat": "Qwen/Qwen1.5-7B-Chat",
        "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
        "Qwen1.5-72B-Chat": "Qwen/Qwen1.5-72B-Chat",

        "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
        "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
        "baichuan2-7b-chat": "baichuan-inc/Baichuan2-7B-Chat",
        "baichuan2-13b-chat": "baichuan-inc/Baichuan2-13B-Chat",

        "internlm-7b": "internlm/internlm-7b",
        "internlm-chat-7b": "internlm/internlm-chat-7b",
        "internlm2-chat-7b": "internlm/internlm2-chat-7b",
        "internlm2-chat-20b": "internlm/internlm2-chat-20b",

        "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
        "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

        "Yi-34B-Chat": "https://huggingface.co/01-ai/Yi-34B-Chat",

        "agentlm-7b": "THUDM/agentlm-7b",
        "agentlm-13b": "THUDM/agentlm-13b",
        "agentlm-70b": "THUDM/agentlm-70b",

        "falcon-7b": "tiiuae/falcon-7b",
        "falcon-40b": "tiiuae/falcon-40b",
        "falcon-rw-7b": "tiiuae/falcon-rw-7b",

        "aquila-7b": "BAAI/Aquila-7B",
        "aquilachat-7b": "BAAI/AquilaChat-7B",
        "open_llama_13b": "openlm-research/open_llama_13b",
        "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
        "koala": "young-geng/koala",
        "mpt-7b": "mosaicml/mpt-7b",
        "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
        "mpt-30b": "mosaicml/mpt-30b",
        "opt-66b": "facebook/opt-66b",
        "opt-iml-max-30b": "facebook/opt-iml-max-30b",
        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",
        "gpt-j-6b": "EleutherAI/gpt-j-6b",
        "gpt4all-j": "nomic-ai/gpt4all-j",
        "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
        "pythia-12b": "EleutherAI/pythia-12b",
        "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "dolly-v2-12b": "databricks/dolly-v2-12b",
        "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    },

    "reranker": {
        "bge-reranker-large": "BAAI/bge-reranker-large",
        "bge-reranker-base": "BAAI/bge-reranker-base",
    }
}

# 通常情况下不需要更改以下内容

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# 使用VLLM可能导致模型推理能力下降，无法完成Agent任务
VLLM_MODEL_DICT = {
    "chatglm2-6b": "THUDM/chatglm2-6b",
    "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

    "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",

    "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
    "baichuan2-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan2-13b-chat": "baichuan-inc/Baichuan-13B-Chat",

    "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
    "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

    "internlm-7b": "internlm/internlm-7b",
    "internlm-chat-7b": "internlm/internlm-chat-7b",
    "internlm2-chat-7b": "internlm/Models/internlm2-chat-7b",
    "internlm2-chat-20b": "internlm/Models/internlm2-chat-20b",

    "aquila-7b": "BAAI/Aquila-7B",
    "aquilachat-7b": "BAAI/AquilaChat-7B",

    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-rw-7b": "tiiuae/falcon-rw-7b",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "gpt4all-j": "nomic-ai/gpt4all-j",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
    "koala": "young-geng/koala",
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
    "mpt-30b": "mosaicml/mpt-30b",
    "opt-66b": "facebook/opt-66b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",

}

SUPPORT_AGENT_MODEL = [
    "openai-api",  # GPT4 模型
    "qwen-api",  # Qwen Max模型
    "zhipu-api",  # 智谱AI GLM4模型
    "Qwen",  # 所有Qwen系列本地模型
    "chatglm3-6b",
    "internlm2-chat-20b",
    "Orion-14B-Chat-Plugin",
]
