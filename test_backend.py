from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from typing import *
import os
# from Script.create_db import DBconnecter
from Script.data_base.kb_service import DBService
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM
# from utils import xuanji_api

UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()

def xuanji_api(prompt, model, intention):
    return "this is resp"

class Message(BaseModel):
    text: str
    model_name: str
    message: list

class Basefile(BaseModel):
    file_names: List[str]
    files: List[UploadFile]

class PromptBase(BaseModel):
    tabel_name: Union[str, None]
    prompt_id: Union[int, None]
    keyword: Union[str, None]
    domain_name: Union[str, None]
    task_name: Union[str, None]
    cls_name: Union[str, None]
    # model_type: Union[str, None]
    prompt: Union[str, None]
    args: Union[str, None]
    query_prompt: Union[str, None]
    answer_prompt: Union[str, None]
    evaluate_prompt: Union[str, None]
    query_args: Union[str, None]
    answer_args: Union[str, None]
    evaluate_args: Union[str, None]

class PromptList(BaseModel):
    prompt_id: Union[int, None]
    tabel_name: Union[str, None]

class SFTDataBase(BaseModel):
    uniqueId: Union[int, None]
    inputs: Union[list, None]
    targets: Union[list, None]
    turn: Union[int, None]
    domain_name: Union[str, None]
    task_name: Union[str, None]
    cls_name: Union[str, None]
    prompt: Union[str, None]
    score: Union[int, None]
    history: Union[str, None]

model = None
tokenizer = None

# @app.post("/load_model/")
# async def load_model():
#     global model, tokenizer
#     if model is None and tokenizer is None:
#         config = AutoConfig.from_pretrained('LLM_model/Llama-2-7b-hf')
#         tokenizer = AutoTokenizer.from_pretrained('LLM_model/Llama-2-7b-hf')
#         model = LlamaForCausalLM.from_pretrained('LLM_model/Llama-2-7b-hf')
#         return {"status": "Model loaded successfully"}
#     else:
#         return {"status": "Model is already loaded"}

# @app.post("/chat/")
# async def chat(message: Message):
#     # 这里可以替换成更复杂的对话逻辑
#     # config = AutoConfig.from_pretrained('LLM_model/Llama-2-7b-hf')
#     # tokenizer = AutoTokenizer.from_pretrained('LLM_model/Llama-2-7b-hf')
#     # model = LlamaForCausalLM.from_pretrained('LLM_model/Llama-2-7b-hf')
    
#     text = f"{message.text}"
#     inputs = tokenizer(text, return_tensors="pt")
    
#     generate_ids = model.generate(inputs.input_ids, max_length=200)
#     outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
#     return {"response": outputs}


@app.post("/chat/")
async def chat(message: Message):
    
    text = f"{message.text}"
    # inputs = tokenizer(text, return_tensors="pt")
    print(message.message)
    resp = xuanji_api(prompt=text, model=message.model_name, intention=True)
    
    # generate_ids = model.generate(inputs.input_ids, max_length=200)
    # outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return {"response": resp}

# Construct Data API
@app.post("/upload/")
async def upload(files: List[UploadFile]):
    all_json_data = []
    results = []
    for file in files:
        results.append(file.filename)

        # 读取文件内容并解析为 JSON
        file_content = await file.read()
        file_json_lines = file_content.decode("utf-8").splitlines()
        json_data = [json.loads(line) for line in file_json_lines]
        all_json_data.append(json_data)

    return {"filenames": results, "json_data": all_json_data}

# Prompt_Base API
@app.post('/prompt_upload/')
async def prompt_upload(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    status = db.insert_data(prompt_data.domain_name, prompt_data.task_name, prompt_data.cls_name, prompt_data.args, prompt_data.query_args, prompt_data.answer_args, prompt_data.evaluate_args, prompt_data.query_prompt, prompt_data.answer_prompt, prompt_data.evaluate_prompt, prompt_data.prompt)
    return {'status': status, 'prompt_data': prompt_data.model_dump()}

@app.post('/prompt_list/')
async def prompt_list(prompt_data: PromptList):
    db = DBService(tabel_name=prompt_data.tabel_name)
    prompt_dict = db.get_all_data()
    return {'data': prompt_dict}
    
@app.post('/prompt_delete/')
async def prompt_delete(prompt_data: PromptList):
    db = DBService(tabel_name=prompt_data.tabel_name)
    resp = db.delete_data(prompt_data.prompt_id)
    return {'status': resp}

@app.post('/prompt_search_keyword/')
async def prompt_search(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    status, prompt_dict = db.get_data_by_keyword(prompt_data.keyword)
    return {'status': status, 'data': prompt_dict}

@app.post('/prompt_update/')
async def prompt_update(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    resp = db.update_data(prompt_data.domain_name, prompt_data.task_name, prompt_data.cls_name, prompt_data.args, prompt_data.query_args, prompt_data.answer_args, prompt_data.evaluate_args, prompt_data.query_prompt, prompt_data.answer_prompt, prompt_data.evaluate_prompt, prompt_data.prompt)
    return {'status': resp, 'prompt_data': prompt_data.model_dump()}

# SftData_base api


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)