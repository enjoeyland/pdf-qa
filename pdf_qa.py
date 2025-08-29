# query_answer.py

from transformers import AutoTokenizer, AutoModel, AutoModelForVision2Seq, Blip2Processor
import torch
import faiss
import numpy as np

from PIL import Image

from utils import extract_pdf_name, embed_texts


# === 텍스트 임베딩 모델 ===
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir="./.cache")
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, cache_dir="./.cache")

# === 멀티모달 QA 모델 (BLIP2) ===
# VISION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
# vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME, cache_dir="./.cache")
# vision_model = AutoModelForVision2Seq.from_pretrained(VISION_MODEL_NAME, cache_dir="./.cache")

class QueryAnswerer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.VEC_STORE_DIR = f"vector_store/{extract_pdf_name(pdf_path)}"
        self.index = faiss.read_index(f"{self.VEC_STORE_DIR}/index.faiss")
        with open(f"{self.VEC_STORE_DIR}/chunks.npy", "rb") as f:
            self.chunks = np.load(f, allow_pickle=True)
    
    def answer_query(self, query: str, top_k: int = 3) -> str:
        # 1. 질의 임베딩
        query_vec = embed_texts(text_tokenizer, text_model, [query])

        # 2. 유사 chunk 검색
        _, indices = self.index.search(query_vec, top_k)
        selected_chunks = [self.chunks[i] for i in indices[0]]
        print(f"유사 chunk 검색 완료: {selected_chunks}")

        # 3. 이미지 기반 응답 (우선순위)
        # for chunk in selected_chunks:
        #     if chunk.get("image_path"):
        #         image = Image.open(chunk["image_path"]).convert("RGB")
        #         inputs = vision_processor(images=image, text=query, return_tensors="pt")
        #         with torch.no_grad():
        #             outputs = vision_model.generate(**inputs)
        #         answer = vision_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #         return answer

        # 4. fallback: 텍스트 chunk 그대로 반환
        if not selected_chunks:
            return "No relevant information found."
        return selected_chunks[0]["text"]

if __name__ == "__main__":
    PDF_PATH = f"pdf/1.제품소개_에스피반도체통신.pdf"

    qa = QueryAnswerer(PDF_PATH)
    print("PDF 기반 QA 시스템입니다.")
    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        answer = qa.answer_query(query)
        print(f"\n🧠 답변: {answer}")