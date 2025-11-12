from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from underthesea import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Text Summarization Demo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Extractive: PhoBERT
extractive_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
extractive_model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
extractive_model.eval()

from peft import PeftModel
import os
import json
import tempfile
import shutil

# Abstractive: ViT5 with LoRA adapter
base_model_name = "VietAI/vit5-base"
lora_adapter_path = r"D:\WORKSPACE\nlp\results\t5_new_lora\checkpoint-1865"

abstractive_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)

# Fix incompatible config parameters
config_file = os.path.join(lora_adapter_path, "adapter_config.json")
with open(config_file, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

# Keep only compatible LoRA config parameters
compatible_params = {
    'r': config_data.get('r'),
    'lora_alpha': config_data.get('lora_alpha'),
    'lora_dropout': config_data.get('lora_dropout'),
    'target_modules': config_data.get('target_modules'),
    'bias': config_data.get('bias'),
    'task_type': config_data.get('task_type'),
    'inference_mode': config_data.get('inference_mode', True),
    'base_model_name_or_path': config_data.get('base_model_name_or_path'),
    'peft_type': config_data.get('peft_type'),
    'fan_in_fan_out': config_data.get('fan_in_fan_out', False),
    'init_lora_weights': config_data.get('init_lora_weights', True)
}
config_data = compatible_params

# Create temp directory and save cleaned config
temp_dir = tempfile.mkdtemp()
temp_config = os.path.join(temp_dir, "adapter_config.json")
with open(temp_config, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=2)

# Copy adapter weights to temp directory
shutil.copy(os.path.join(lora_adapter_path, "adapter_model.safetensors"), 
            os.path.join(temp_dir, "adapter_model.safetensors"))

# Load model from temp directory with cleaned config
abstractive_model = PeftModel.from_pretrained(base_model, temp_dir).to(device)
abstractive_model.eval()

# Clean up temp directory
shutil.rmtree(temp_dir)

class SummarizeRequest(BaseModel):
    text: str
    mode: str  # "extractive", "abstractive", or "both"
    top_k: int = None  # Optional: number of sentences for extractive summarization

def sentence_embedding(sent):
    tokens = extractive_tokenizer(sent, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = extractive_model(**tokens)
    cls_vec = out.last_hidden_state[:,0,:]
    return cls_vec.squeeze(0).cpu().numpy()

def doc_sentence_vectors(text):
    sentences = sent_tokenize(text)
    vecs = [sentence_embedding(s) for s in sentences]
    return sentences, np.vstack(vecs)

def mmr(sentences, sentence_vecs, top_k=1, lambda_=0.7):
    doc_vec = sentence_vecs.mean(axis=0, keepdims=True)
    selected = []
    selected_indices = []

    sim_to_doc = cosine_similarity(sentence_vecs, doc_vec).flatten()

    for _ in range(top_k):
        if len(selected)==0:
            idx = np.argmax(sim_to_doc)
            selected.append(sentences[idx])
            selected_indices.append(idx)
        else:
            mmr_score = []
            for i in range(len(sentences)):
                if i in selected_indices: 
                    mmr_score.append(-999)
                    continue
                sim_i_doc = sim_to_doc[i]
                sim_i_selected = max(cosine_similarity(sentence_vecs[i].reshape(1,-1), 
                                                       sentence_vecs[selected_indices]).flatten())
                score = lambda_ * sim_i_doc - (1-lambda_)*sim_i_selected
                mmr_score.append(score)
            idx = np.argmax(mmr_score)
            selected.append(sentences[idx])
            selected_indices.append(idx)

    selected_indices.sort()
    return " ".join([sentences[i] for i in selected_indices])

def topk_by_avg_tokens(sentences, max_output_tokens=128, tokenizer=extractive_tokenizer, safety_min_k=1):
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]
    avg = (sum(lengths) / len(lengths)) if lengths else 1.0
    k = max(safety_min_k, round(max_output_tokens / avg))
    return k

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    if request.mode not in ["extractive", "abstractive", "both"]:
        raise HTTPException(status_code=400, detail="Mode must be 'extractive', 'abstractive', or 'both'")
    
    result = {}
    
    if request.mode in ["extractive", "both"]:
        sents, vecs = doc_sentence_vectors(request.text)
        # Use top_k from request if provided, otherwise calculate automatically
        if request.top_k is not None:
            top_k = max(1, min(request.top_k, len(sents)))  # Ensure top_k is valid
        else:
            top_k = topk_by_avg_tokens(sents, 128)
        result["extractive"] = mmr(sents, vecs, top_k=top_k)
    
    if request.mode in ["abstractive", "both"]:
        inputs = abstractive_tokenizer("summarize: " + request.text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            summary_ids = abstractive_model.generate(**inputs, max_new_tokens=128, num_beams=4, length_penalty=2.0, early_stopping=True)
        result["abstractive"] = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return result

@app.get("/")
async def root():
    return {"message": "Text Summarization Demo API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)