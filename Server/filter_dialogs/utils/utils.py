import torch

from Server.model_workers.base import LLMModelBase

def single_ppl(llm_model: LLMModelBase, prompt, output=None):
    ppl = llm_model.get_loss(prompt=prompt, output=output)
    ppl = torch.exp(ppl)
    ppl_r = ppl.numpy().item()
    return ppl_r

# For LESS's InfAdam Score
def get_data_statistics(lm_datasets):
    """ 
    Get the data statistics of the dataset. 
    其中 lm_datasets [input_ids, labels, attention_masks]
    """
    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]

        completion_lens = []
        for labels in examples["labels"]:
            com_len = (torch.tensor(labels) > -1).sum()
            completion_lens.append(com_len)
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key in lm_datasets:
        dataset = lm_datasets[key]
        data_size = len(dataset)
        dataset = dataset.map(get_length, batched=True)
        lengths = dataset["length"]
        length = sum(lengths) / len(lengths)
        c_lengths = dataset["c_length"]
        c_length = sum(c_lengths) / len(c_lengths)
        print(
            f"[{key} set] examples: {data_size}; # avg tokens: {length}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {c_length}")
        
def add_padding_to_tokenizer(tokenizer):
    """ add the padding tokens in the tokenizer """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


# Get train dataset
def encode_with_messages_format(example, tokenizer, max_seq_length):
        '''
        Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        '''
        messages = example['messages']
        if len(messages) == 0:
            raise ValueError('messages field is empty.')

        example_text = concat_messages(messages, tokenizer)
        tokenized_example = tokenizer(
            example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = concat_messages(
                        messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
                else:
                    messages_so_far = concat_messages(
                        messages[:message_idx+1], tokenizer)
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt',
                    max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100

                if message_end_idx >= max_seq_length:
                    break

        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
