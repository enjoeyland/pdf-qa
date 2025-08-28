# document_indexer.py
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss
import os
import torch
import json, re
import openai

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pdf_analyzer import extract_pdf_content
from utils import extract_pdf_name, embed_texts

# === 설정 ===
PDF_PATH = f"pdf/1.제품소개_에스피반도체통신.pdf"
VEC_STORE_DIR = f"vector_store/{extract_pdf_name(PDF_PATH)}"
os.makedirs(VEC_STORE_DIR, exist_ok=True)


# === 모델 로딩 ===
llm_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Mistral-7B-DPO", cache_dir="./.cache", use_fast=True) # pip install sentencepiece
llm_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./.cache"
)

embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir="./.cache")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir="./.cache")

# === 의미 단위 청크 생성 ===
def build_chunk_prompt(page_blocks):
    page_text = [{"text": block["text"], "bbox": tuple(round(x, 1) for x in block["bbox"])} for block in page_blocks if block["text"].strip()]
    return f"""다음은 텍스트와 위치를 나타내, 나누어져 있는 텍스트를 의미 단위로 묶어서 텍스트로 답변해줘. Json의 list[str] 형태로 답변해줘. 텍스트는 수정하지 말아줘:\n\n{page_text}\n"""

def chunk_with_llm(page_blocks):
    prompt = build_chunk_prompt(page_blocks)
    print(f"청크 생성 프롬프트:\n{prompt}\n")
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=1024)
    decoded = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(f"청크 생성 결과:\n{decoded}\n")
    match = re.search(r"\[\s*\".*?\"\s*\]", decoded, re.DOTALL)
    if match:
        print(f"청크 추출 결과: {match.group()}")
        return json.loads(match.group())
    return [decoded]


def chunk_with_llm_via_api(page_blocks, model="gpt-5-mini"):
    # page_blocks → 텍스트 합치지 않고 block별 유지된 리스트
    prompt = build_chunk_prompt(page_blocks)
    print(f"[API 프롬프트]\n{prompt}\n")

    response = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
        }],
        response_format={
            "type": "json_object"
        },
        # verbosity="medium",
        reasoning_effort="medium",
        store=True
    )
    # response = openai.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.7,
    #     max_tokens=1024,
    #     user="pdf-qa-system",
    #     reasoning={"effort": "medium"}
    # )

    decoded = response.choices[0].message.content.strip()
    print(f"[API 반환 텍스트]\n{decoded}\n")

    match = re.search(r"\[\s*\".*?\"\s*\]", decoded, re.DOTALL)
    if match:
        print(f"[추출된 JSON 청크]\n{match.group()}\n")
        return json.loads(match.group())

    print("[⚠️ JSON 청크 추출 실패] 디코딩 전체 반환")
    return [decoded]


# === FAISS 인덱스 생성 ===
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# === 전체 파이프라인 실행 ===
all_chunks = []
page_blocks = extract_pdf_content(PDF_PATH)  # List[List[block]]

for blocks in page_blocks:
    chunks = chunk_with_llm_via_api(blocks)
    print(f"추출된 청크 수: {len(chunks)}")
    for chunk in chunks:
        all_chunks.append({"text": chunk})  # 위치정보를 넣고 싶다면 연결 필요

embeddings = embed_texts(embed_tokenizer, embed_model, [c["text"] for c in all_chunks])
index = build_faiss_index(embeddings)

faiss.write_index(index, os.path.join(VEC_STORE_DIR, "index.faiss"))
np.save(os.path.join(VEC_STORE_DIR, "chunks.npy"), np.array(all_chunks, dtype=object))
print(f"✅ 벡터 DB 저장 완료: {len(all_chunks)} chunks")










# # === 1. PDF 분석 ===
