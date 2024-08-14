from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaForCausalLM
import torch

app = FastAPI()

class Message(BaseModel):
    text: str

model = None
tokenizer = None

@app.post("/load_model/")
async def load_model():
    global model, tokenizer
    if model is None and tokenizer is None:
        config = AutoConfig.from_pretrained('LLM_model/Llama-2-7b-hf')
        tokenizer = AutoTokenizer.from_pretrained('LLM_model/Llama-2-7b-hf')
        model = LlamaForCausalLM.from_pretrained('LLM_model/Llama-2-7b-hf')
        return {"status": "Model loaded successfully"}
    else:
        return {"status": "Model is already loaded"}

@app.post("/chat/")
async def chat(message: Message):
    # 这里可以替换成更复杂的对话逻辑
    # config = AutoConfig.from_pretrained('LLM_model/Llama-2-7b-hf')
    # tokenizer = AutoTokenizer.from_pretrained('LLM_model/Llama-2-7b-hf')
    # model = LlamaForCausalLM.from_pretrained('LLM_model/Llama-2-7b-hf')
    
    text = f"{message.text}"
    inputs = tokenizer(text, return_tensors="pt")
    
    generate_ids = model.generate(inputs.input_ids, max_length=200)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return {"response": outputs}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)