# document_indexer.py
from dotenv import load_dotenv
load_dotenv()

import os
import json, re
import numpy as np
import faiss
from datetime import datetime

from pdf_analyzer import extract_pdf_content
from utils import extract_pdf_name
from models_factory import EmbeddingModel, LLM, create_embedding, create_llm


# === 의미 단위 청크 생성 ===
def build_chunk_prompt(page_blocks):
    page_text = [
        {"idx": i, "text": block["text"], "bbox": tuple(round(x, 1) for x in block["bbox"])} 
        for i, block in enumerate(page_blocks) if block.get("text","").strip()
    ]
    return (
        "다음은 PDF에서 추출한 텍스트 블록과 위치 정보입니다.\n"
        "각 블록에는 고유한 'idx' 값이 있습니다.\n"
        "의미 단위로 블록들을 묶어서 반환해주세요. "
        "텍스트는 수정하지 말고, 원문 그대로 사용해야 합니다. "
        "단, 표의 경우 각 제품 별로 하나의 텍스트를 작성해주세요. \n\n"
        "출력 형식은 JSON 배열이며, 각 항목은 아래와 같은 구조여야 합니다:\n"
        "[\n"
        "  {\"text\": \"묶인 전체 텍스트\", \"idx\": [해당 블록들의 idx 리스트]},\n"
        "  {\"text\": \"다음 묶인 텍스트\", \"idx\": [ ... ]}\n"
        "]\n\n"
        f"입력 블록:\n{"\n".join([str(block) for block in page_text])}\n"
    )
    # return (
    #     "다음은 텍스트와 위치를 나타냅니다.\n"
    #     "나누어져 있는 텍스트를 의미 단위로 묶어서 반환해주세요. "
    #     "표의 경우 한 제품별로 묶어주세요. "
    #     "텍스트는 수정하지 말아주세요. "
    #     "Json list[str] 형태로만 \n\n"
    #     f"{page_text}\n"
    # )
def parse_chunk_output(out: str):
    """
    LLM 출력에서 청크 리스트를 추출한다.
    기대 형식:
    {
      "result": [
        {"text": "...", "idx": [0,1,2]},
        {"text": "...", "idx": [3,4]}
      ]
    }

    반환: List[dict] (각 dict는 {"text": str, "idx": List[int]})
    실패 시 None 반환
    """
    out = out.strip()

    # 먼저 JSON 전체 파싱 시도
    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict) and "result" in parsed:
            if isinstance(parsed["result"], list):
                assert all(isinstance(item, dict) and "text" in item and "idx" in item for item in parsed["result"])
                return parsed["result"]
        if isinstance(parsed, list):
            assert all(isinstance(item, dict) and "text" in item and "idx" in item for item in parsed)
            return parsed
        if isinstance(parsed, dict) and "text" in parsed and "idx" in parsed:
            return [parsed]
    except Exception:
        pass

    raise Exception("LLM 출력에서 청크 추출 실패")   

class ChunkExtractionError(Exception):
    def __init__(self, prompt: str, raw_outputs: list):
        self.prompt = prompt
        self.raw_outputs = raw_outputs
        super().__init__(f"LLM JSON 추출 실패: {len(self.raw_outputs)}번 시도 모두 실패")
        # super().__init__(f"LLM JSON 추출 실패\nPrompt:\n{prompt}\n\nOutput:\n{raw_output}")

def chunk_with_llm(llm, page_blocks):
    # page_blocks → 텍스트 합치지 않고 block별 유지된 리스트
    prompt = build_chunk_prompt(page_blocks)
    print(f"[API 프롬프트]\n{prompt}\n")

    out = llm.generate(prompt)
    print(f"[API 반환 텍스트]\n{out}\n")

    try:
        chunks = parse_chunk_output(out)
        return chunks
    except Exception:
        raise ChunkExtractionError(prompt, [out])

    # match = re.search(r"\[\s*\".*?\"\s*\]", out, re.DOTALL)
    # if match:
    #     print(f"[추출된 JSON 청크]\n{match.group()}\n")
    #     return json.loads(match.group())

    # raise ChunkExtractionError(prompt, [out])

def extract_chunks_with_retry(llm, blocks, try_count: int = 3):
    """
    여러 번 재시도하면서, 실패 시마다 에러를 기록한다.
    - 성공 시 → chunks 반환
    - 실패 시 → ChunkExtractionError(errors=[...]) 발생
    """
    errors = []
    for attempt in range(1, try_count + 1):
        try:
            chunks = chunk_with_llm(llm, blocks)
            print(f"✅ 추출 성공 (시도 {attempt}/{try_count}) → {len(chunks)}개 청크")
            return chunks
        except ChunkExtractionError as e:
            print(f"❌ 추출 실패 (시도 {attempt}/{try_count})\n")
            errors.append({
                "attempt": attempt,
                "prompt": e.prompt,
                "raw_outputs": e.raw_outputs
            })

    # 모든 시도 실패 → 누적된 에러 포함해서 반환
    raise ChunkExtractionError(
        prompt=errors[-1]["prompt"] if errors else "",
        raw_outputs=[e["raw_outputs"] for e in errors]
    )

def get_bbox(blocks, idxs):
    (x0, y0, x1, y1) = (float('inf'), float('inf'), float('-inf'), float('-inf'))
    for idx, block in enumerate(blocks):
        if idx in idxs:
            (x0, y0, x1, y1) = (
                min(x0, block["bbox"][0]),
                min(y0, block["bbox"][1]),
                max(x1, block["bbox"][2]),
                max(y1, block["bbox"][3]),
            )
    return (x0, y0, x1, y1)

# === FAISS 인덱스 생성 ===
def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])  # 코사인 동치
    index.add(embeddings)
    return index

# === 전체 파이프라인 실행 ===
def build_pdf_faiss_index(pdf_path: str, embedder: EmbeddingModel, llm: LLM):
    page_blocks = extract_pdf_content(pdf_path)  # List[List[block]]

    all_chunks = []
    failed_cases = []
    for i, blocks in enumerate(page_blocks):
        try:
            chunks = extract_chunks_with_retry(llm, blocks, try_count=2)
            for chunk in chunks:
                all_chunks.append({
                    "page": i + 1,
                    "text": chunk["text"],
                    "bbox": get_bbox(blocks, chunk["idx"])
                })

        except ChunkExtractionError as e:
            print(f"[page {i+1}] ❌ 모든 재시도 실패")
            failed_cases.append({
                "page": i + 1,
                "prompt": e.prompt,
                "raw_outputs": e.raw_outputs,
            })

    embeddings = embedder.encode([c["text"] for c in all_chunks], is_query=False)  # L2 정규화됨
    index = build_faiss_index(embeddings)


    VEC_STORE_DIR = f"vector_store/{extract_pdf_name(pdf_path)}"
    os.makedirs(VEC_STORE_DIR, exist_ok=True)
    with open(os.path.join(VEC_STORE_DIR, "chunks_history.jsonl"), "a") as f:
        log = {
            "time": datetime.now().isoformat(),
            "source_pdf": pdf_path,
            "num_chunks": len(all_chunks),
            "chunks": all_chunks,
            "failure": failed_cases
        }
        f.write(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    faiss.write_index(index, os.path.join(VEC_STORE_DIR, "index.faiss"))
    np.save(os.path.join(VEC_STORE_DIR, "chunks.npy"), np.array(all_chunks, dtype=object))
    print(f"✅ 벡터 DB 저장 완료: {len(all_chunks)} chunks")


if __name__ == "__main__":
    # === 설정 ===
    EMB_NAME = "kure"
    LLM_NAME = "gpt-5-mini"
    PDF_PATH = f"pdf/1.제품소개_에스피반도체통신.pdf"

    # === 모델 로딩 ===
    embedder = create_embedding(EMB_NAME)
    llm = create_llm(LLM_NAME)

    build_pdf_faiss_index(PDF_PATH, embedder, llm)
