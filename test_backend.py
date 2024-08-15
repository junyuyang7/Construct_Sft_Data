from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from typing import *
import os
# from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM
# from utils import xuanji_api

UPLOAD_DIRECTORY = "./uploaded_files"

app = FastAPI()

def xuanji_api(prompt, model, intention):
    return "this is resp"

class Message(BaseModel):
    text: str
    model_name: str
    message: list

class Basefile(BaseModel):
    text: str
    model_name: str
    message: list

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

@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    for file in files:
        # 保存上传的文件到指定目录
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

    return {"filenames": [file.filename for file in files]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)