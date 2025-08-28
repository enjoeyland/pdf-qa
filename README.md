
# 📘 PDF QA with RAG

PDF 문서를 **검색 가능(Searchable)한 지식 베이스**로 만들어,  
사용자의 질의(Query)에 대해 **LLM + 벡터 검색(RAG: Retrieval-Augmented Generation)** 기반으로 답변하는 시스템입니다.



## 🚀 요구사항

- **Searchable PDF** (텍스트 추출 가능한 PDF)
- **OpenAI API Key** (환경 변수 `OPENAI_API_KEY` 로 설정)



## 🧭 방법론

### 1. 준비 단계

1. **PDF에서 데이터 추출**
   - `PyMuPDF`를 활용하여 PDF에서 텍스트와 이미지를 추출
   - 위치 정보(`bbox`) 포함
   - 예시:
     ```python
     import fitz
     doc = fitz.open(pdf_path)
     for page_idx, page in enumerate(doc):
         blocks = page.get_text("dict")["blocks"]
         img_infos = page.get_images(full=True)
     ```

2. **추출한 데이터를 LLM을 통해 의미 있는 정보로 정제**
   - OpenAI API (`gpt-5-mini`)를 활용하여 줄 단위 텍스트를 의미 단위 문단으로 묶음
   - 예시 프롬프트:
     ```python
     def build_chunk_prompt(page_blocks):
         page_text = [
             {"text": block["text"], "bbox": tuple(round(x,1) for x in block["bbox"])}
             for block in page_blocks if block["text"].strip()
         ]
         return f"""
         다음은 텍스트와 위치를 나타내,
         나누어져 있는 텍스트를 의미 단위로 묶어서 텍스트로 답변해줘.
         Json의 list[str] 형태로 답변해줘.
         텍스트는 수정하지 말아줘:

         {page_text}
         """
     ```

3. **PDF 정보를 Encoder 모델을 통해 임베딩 후 벡터 스토어에 저장**
   - `sentence-transformers/all-MiniLM-L6-v2` 사용
   - FAISS 인덱스 구축
   - 예시:
     ```python
     def embed_texts(texts):
         inputs = embed_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
         with torch.no_grad():
             outputs = embed_model(**inputs)
         embeddings = mean_pooling(outputs, inputs["attention_mask"])
         return embeddings.cpu().numpy()

     def build_faiss_index(embeddings):
         index = faiss.IndexFlatL2(embeddings.shape[1])
         index.add(embeddings)
         return index
     ```

---

### 2. 추론 단계

1. 사용자로부터 `query` 입력 받기  
2. Query를 임베딩하여 FAISS 벡터 스토어에서 유사 chunk 검색  
3. 검색된 정보를 바탕으로 LLM이 최종 답변 생성  

```python
def answer_query(pdf_path: str, query: str, top_k: int = 3) -> str:
    VEC_STORE_DIR = f"vector_store/{extract_pdf_name(pdf_path)}"

    # 1. 인덱스 + chunks 불러오기
    index = faiss.read_index(f"{VEC_STORE_DIR}/index.faiss")
    with open(f"{VEC_STORE_DIR}/chunks.npy", "rb") as f:
        chunks = np.load(f, allow_pickle=True)

    # 2. 질의 임베딩
    query_vec = embed_texts([query])

    # 3. 유사 chunk 검색
    _, indices = index.search(query_vec, top_k)
    selected_chunks = [chunks[i] for i in indices[0]]

    return selected_chunks
````

## 📂 프로젝트 구조

```
pdf-qa/
├── document_indexer.py     # PDF → 청크 생성, 임베딩, 벡터스토어 저장
├── pdf_analyzer.py         # PDF 텍스트/이미지 추출
├── query_answer.py         # 질의 응답 로직 (FAISS 검색 + LLM 호출)
├── pdf_qa.py               # CLI 실행 진입점
│
├── utils.py                # 공용 유틸 함수 (extract_pdf_name, embed_texts 등)
│
├── vector_store/           # PDF별 벡터 인덱스 및 청크 저장
├── extracted_images/       # PDF에서 추출된 이미지 저장
├── .cache/                 # 모델 캐시 (HuggingFace, OpenAI)
│
├── .env                    # 환경 변수 (OPENAI_API_KEY 등)
├── .gitignore              # Git 관리 제외 파일 정의
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 문서
```

## 🛠 설치 및 실행

```bash
# 1. 의존성 설치
# PyTorch 공식 설치 가이드에서 확인 후 torch, torchvision 설치: https://pytorch.org/get-started/locally/
pip install -r requirements.txt

# 2. 환경변수 설정
# 프로젝트 루트에 .env 파일 생성 후 아래 내용 추가
# OPENAI_API_KEY=your_api_key_here

# 3. PDF 인덱싱
python document_indexer.py

# 4. 질의응답 실행
python pdf_qa.py
```
