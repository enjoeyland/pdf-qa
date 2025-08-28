import os
import torch

def extract_pdf_name(pdf_path: str) -> str:
    """
    PDF 경로에서 .pdf 확장자를 가진 파일 이름(확장자 제외)만 추출.

    예:
        pdf_path = "pdf/청년창업센터.pdf" → "청년창업센터"
        pdf_path = "pdf/문서.txt" → ValueError 발생
    """
    filename = os.path.basename(pdf_path)            # "청년창업센터.pdf"
    name, ext = os.path.splitext(filename)           # ("청년창업센터", ".pdf")

    if ext.lower() != ".pdf":
        raise ValueError(f"지원되지 않는 파일 확장자입니다: {ext} (PDF만 허용)")

    return name


def embed_texts(tokenizer, model, texts):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])
    return embeddings.cpu().numpy()