# query_answer.py
from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np

from models_factory import EmbeddingModel, LLM, create_embedding, create_llm
from utils import extract_pdf_name

# === 멀티모달 QA 모델 (BLIP2) ===
# VISION_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
# vision_processor = Blip2Processor.from_pretrained(VISION_MODEL_NAME, cache_dir="./.cache")
# vision_model = AutoModelForVision2Seq.from_pretrained(VISION_MODEL_NAME, cache_dir="./.cache")

class QueryAnswerer:
    def __init__(self, pdf_path: str, embedder: EmbeddingModel, llm: LLM):
        self.pdf_path = pdf_path
        self.embedder = embedder
        self.llm = llm
        self.VEC_STORE_DIR = f"vector_store/{extract_pdf_name(pdf_path)}"
        self.index = faiss.read_index(f"{self.VEC_STORE_DIR}/index.faiss")
        with open(f"{self.VEC_STORE_DIR}/chunks.npy", "rb") as f:
            self.chunks = np.load(f, allow_pickle=True)
    
    def get_relevant_chunks(self, query: str, top_k: int = 3) -> str:
        # 1. 질의 임베딩
        query_vec = self.embedder.encode([query], is_query=True)

        # 2. 유사 chunk 검색
        _, indices = self.index.search(query_vec, top_k)
        selected_chunks = [self.chunks[i] for i in indices[0]]
        print(f"유사 chunk 검색 완료: {selected_chunks}")
        return selected_chunks


    def _build_prompt(self, query: str, chunks: list) -> str:
        context = "\n\n".join([f"Context {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks)])
        prompt = (
            "다음은 PDF 문서에서 추출한 내용입니다. "
            "이 정보를 바탕으로 질문에 답변해 주세요. "
            "모르는 내용은 'No relevant information found.'라고 답변해 주세요.\n\n"
            f"{context}\n\n"
            f"질문: {query}\n"
            "답변:"
        )
        return prompt
    
    def answer_query(self, query: str) -> str:
        selected_chunks = self.get_relevant_chunks(query, top_k=4)

        # 2. LLM 응답 생성
        prompt = self._build_prompt(query, selected_chunks)
        answer = self.llm.generate(prompt)
        return answer

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
    EMB_NAME = "kure"
    LLM_NAME = "gpt-5-mini"

    embedder = create_embedding(EMB_NAME)
    llm = create_llm("custom_openai", model_id=LLM_NAME, response_format="text")
    qa = QueryAnswerer(PDF_PATH, embedder, llm)

    print("PDF 기반 QA 시스템입니다.")
    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        answer = qa.answer_query(query)
        print(f"\n🧠 답변: {answer}")
    
    # Sample queries:
    # 1. 회사는 어떤 환경방침을 준수하고 있나요?
    # 2. TO-3P 제품의 특징은 무엇인가요?
    # 3. 2023년에는 어떤 사건이 있었나요?
    # 4. 회사 이름이 어떻게 되나요?
    # 5. Saw 공정에 대해서 설명해줘.
    # 6. 어떤 품질을 강조하고 있어?
    # 7. TO-252 HV 제품의 특징은 무엇인가요? 
    #   -> 표 병합에 따라서 문제가 있다.

    # indexing(gpt-5-mini, kure), retrieve(kure),  generation(gpt-5-mini)
    # indexing      → (1: O, 2: O, 3: O, 4: O, 5: O, 6: △, 7: X)
    # retrieve      → (1: O, 2: O, 3: O, 4: X, 5: O, 6: △, 7: O)
    # generation    → (1: O, 2: O, 3: △, 4: O, 5: O, 6: △, 7: O)