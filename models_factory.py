# models_factory.py

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Literal

# ===== Embedding Factory =====
class EmbeddingConfig(dict):
    def __init__(self, model_id: str, cache_dir="./.cache", **kwargs):
        super().__init__(model_id=model_id, cache_dir=cache_dir, **kwargs)

    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value
    
    def to_kwargs(self):
        return {k: v for k, v in self.items() if k not in ("model_id", "normalize", "use_e5_prefix")}

class EmbeddingModel:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.token = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
        self.model = AutoModel.from_pretrained(cfg.model_id, **cfg.to_kwargs())

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.cfg.use_e5_prefix:
            pref = "query: " if is_query else "passage: "
            texts = [pref + t for t in texts]
        x = self.token(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**x).last_hidden_state
        emb = self._mean_pool(out, x["attention_mask"]).cpu().numpy().astype("float32")
        if self.cfg.normalize:
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        return emb

# ===== LLM Factory =====
class LLMConfig(dict):
    def __init__(self, backend: Literal["hf", "openai"], model_id: str, cache_dir="./.cache", **kwargs):
        if backend not in ("hf", "openai"):
            raise ValueError(f"Unsupported backend: {backend}. Use 'hf' or 'openai'.")
        super().__init__(backend=backend, model_id=model_id, cache_dir=cache_dir, **kwargs)
        self._from_pretrained_keys  = {"cache_dir"}
        self._generate_keys = {"max_new_tokens", "temperature", "top_p", "top_k", "do_sample"}

    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value
    
    def to_from_pretrained_kwargs(self):
        return {k: v for k, v in self.items() if k in self._from_pretrained_keys}
    
    def to_generate_kwargs(self):
        return {k: v for k, v in self.items() if k in self._generate_keys}

class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if cfg.backend == "hf":
            self.token = AutoTokenizer.from_pretrained(cfg.model_id, cache_dir=cfg.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **cfg.to_from_pretrained_kwargs())

    def generate(self, prompt: str) -> str:
        if self.cfg.backend == "hf":
            inputs = self.token(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                print({**inputs, "pad_token_id": self.token.eos_token_id, **self.cfg.to_generate_kwargs()})
                out = self.model.generate(**inputs, pad_token_id=self.token.eos_token_id, **self.cfg.to_generate_kwargs())
            return self.token.decode(out[0], skip_special_tokens=True)
        
        elif self.cfg.backend == "openai":
            import openai
            resp = openai.chat.completions.create(
                model=self.cfg.model_id,  # e.g., "gpt-5-mini"
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                response_format={
                    "type": self.cfg.get("response_format", "text")
                },
                reasoning_effort="medium",
                store=True
            )
            return resp.choices[0].message.content

# ===== Presets & Factory Functions =====
def create_embedding(name: str, model_id: str = None, **kwargs) -> EmbeddingModel:
    """name: minilm | kure | e5-base | custom"""
    if name == "custom" and model_id is None:
        raise ValueError("model_id must be specified for custom embeddings.")
    presets = {
        "minilm": {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize": True
        },
        "kure": {
            "model_id": "nlpai-lab/KURE-v1",
            "normalize": True
        },
        "e5-base": {
            "model_id": "intfloat/multilingual-e5-base",
            "normalize": True,
            "use_e5_prefix": True
        },
        "custom": {
            "model_id": model_id,
            "normalize": True
        }
    }
    if name in presets:
        return EmbeddingModel(EmbeddingConfig(**presets[name], **kwargs))

    raise ValueError(f"Unknown embedding preset: {name}")

    

def create_llm(name: str, model_id: str = None, **kwargs) -> LLM:
    """name: nous-hermes | gpt-5-mini | custom_hf | custom_openai"""
    if name in ("custom_hf", "custom_openai") and model_id is None:
        raise ValueError("model_id must be specified for custom LLMs.")
    
    presets = {
        "nous-hermes": {
            "backend": "hf",
            "model_id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
            "max_new_tokens": 1024
        },
        "gpt-5-mini": {
            "backend": "openai",
            "model_id": "gpt-5-mini",
            "response_format": "json_object"
        },
        "custom_hf": {
            "backend": "hf",
            "model_id": model_id
        },
        "custom_openai": {
            "backend": "openai",
            "model_id": model_id,
            "response_format": "json_object"
        }
    }

    if name in presets:
        return LLM(LLMConfig(**(presets[name] | kwargs)))
    
    raise ValueError(f"Unknown LLM preset: {name}")
