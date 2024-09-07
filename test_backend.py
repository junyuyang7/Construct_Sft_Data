from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
import uvicorn
from typing import *
import os
# from Script.create_db import DBconnecter
from Script.data_base.kb_service import DBService
import json
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM

from Script.model_workers.base import LLMModelBase
from Script.config import Args, llm_model_dict
from web_pages.utils import construct_dialog
# from utils import xuanji_api

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
UPLOAD_DIRECTORY = "./uploaded_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app = FastAPI()

def xuanji_api(prompt, model, intention):
    return "this is resp"

class ModelRequest(BaseModel):
    model_name: str

class Message(BaseModel):
    text: str
    message: Union[List[dict], None]

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
    first_query_prompt: Union[str, None]
    answer_prompt: Union[str, None]
    evaluate_prompt: Union[str, None]
    first_query_args: Union[str, None]
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

llm_model = None

def load_model(model_name):
    global llm_model
    
    if llm_model is not None:
        # 释放原有模型的显存
        del llm_model
        torch.cuda.empty_cache()
        
    args = Args()
    args.model_path = llm_model_dict[model_name]
    llm_model = LLMModelBase(args)
    
    return f"Loaded {model_name} successfully"

@app.post("/switch_model/")
async def switch_model(request: Request):
    data = await request.json()  # 解析请求体中的 JSON 数据
    model_name = data.get("model_name")  # 获取 model_name 字段
    if not model_name:
        return {"error": "Model name not provided"}
    
    result = load_model(model_name)
    return {"status": result}

@app.get("/current_model/")
async def current_model(request: Request):
    data = await request.json()  # 解析请求体中的 JSON 数据
    model_name = data.get("model_name")  # 获取 model_name 字段
    if not model_name:
        return {"error": "Model name not provided"}
    return {"current_model": model_name}

@app.post("/chat/")
async def chat(message: Message):
    resp, chat_lst = llm_model.generate(prompt=message.text, chat_lst=message.message)
        
    return {"response": resp, 'chat_lst': chat_lst}

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

@app.post("/construct_dialog")
async def construct_sft_data(request: Request):
    data = await request.json()  # 解析请求体中的 JSON 数据
    final_prompt_lst = data.get("final_prompt_lst")
    model_name = data.get("model_name")
    
    if not model_name or final_prompt_lst:
        return {"error": "Model name not provided"}
    ans_df_lst, filter_prompt_lst = construct_dialog(final_prompt_lst, model_name)
    
    return {'ans_df_lst': ans_df_lst, 'filter_prompt_lst': filter_prompt_lst}

# Prompt_Base API
@app.post('/prompt_upload/')
async def prompt_upload(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    status = db.insert_data(prompt_data.domain_name, 
                            prompt_data.task_name, 
                            prompt_data.cls_name, 
                            prompt_data.args, 
                            prompt_data.first_query_args, 
                            prompt_data.query_args, 
                            prompt_data.answer_args, 
                            prompt_data.evaluate_args, 
                            prompt_data.first_query_prompt, 
                            prompt_data.query_prompt, 
                            prompt_data.answer_prompt, 
                            prompt_data.evaluate_prompt, 
                            prompt_data.prompt)
    return {'status': status, 'prompt_data': prompt_data.model_dump()}

@app.post('/prompt_list/')
async def prompt_list(prompt_data: PromptList):
    db = DBService(tabel_name=prompt_data.tabel_name)
    prompt_dict = db.get_all_data()
    # prompt_dict = [{}, {}, {}, ....] List[dict]
    return {'data': prompt_dict}
    
@app.post('/prompt_delete/')
async def prompt_delete(prompt_data: PromptList):
    db = DBService(tabel_name=prompt_data.tabel_name)
    resp = db.delete_data(prompt_data.prompt_id)
    return {'status': resp}

# TODO
@app.post('/prompt_search_keyword/')
async def prompt_search(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    status, prompt_dict = db.get_data_by_keyword(prompt_data.keyword)
    return {'status': status, 'data': prompt_dict}

@app.post('/prompt_update/')
async def prompt_update(prompt_data: PromptBase):
    db = DBService(tabel_name=prompt_data.tabel_name)
    resp = db.update_data(prompt_data.domain_name, 
                        prompt_data.task_name, 
                        prompt_data.cls_name, 
                        prompt_data.args, 
                        prompt_data.first_query_args, 
                        prompt_data.query_args, 
                        prompt_data.answer_args, 
                        prompt_data.evaluate_args, 
                        prompt_data.first_query_prompt, 
                        prompt_data.query_prompt, 
                        prompt_data.answer_prompt, 
                        prompt_data.evaluate_prompt, 
                        prompt_data.prompt)
    return {'status': resp, 'prompt_data': prompt_data.model_dump()}

# SftData_base api


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)