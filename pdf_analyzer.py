# pdf_analyzer.py

import fitz  # PyMuPDF
import os
from utils import extract_pdf_name

def extract_pdf_content(pdf_path, image_output_dir="extracted_images"):
    pdf_name = extract_pdf_name(pdf_path)
    image_output_dir = os.path.join(image_output_dir, pdf_name)
    os.makedirs(image_output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_blocks = []

    for page_idx, page in enumerate(doc):
        page_items = []

        # 텍스트 블록 추출
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # 텍스트
                text = "\n".join([
                    span["text"] for line in block["lines"] for span in line["spans"]
                ]).strip()
                bbox = block["bbox"]
                if text:
                    page_items.append({
                        "text": text,
                        "bbox": bbox,
                        "page": page_idx + 1,
                        "image_path": None
                    })

        # 이미지 추출
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                image_path = os.path.join(image_output_dir, f"page{page_idx}_img{img_idx}.png")
                pix.save(image_path)
                page_items.append({
                    "text": "",
                    "bbox": None,
                    "page": page_idx + 1,
                    "image_path": image_path
                })
            except Exception as e:
                print(f"[경고] 이미지 추출 실패: {page_idx=}, {img_idx=}")
        page_blocks.append(page_items)
    return page_blocks

if __name__ == "__main__":
    pdf_path = "pdf/1.제품소개_에스피반도체통신.pdf"  # PDF 파일 경로
    page_blocks = extract_pdf_content(pdf_path)
    print(f"추출된 페이지 수: {len(page_blocks)}")
    for page in page_blocks:
        print(page)