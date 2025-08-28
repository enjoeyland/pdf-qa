# pdf_qa.py
from query_answer import answer_query

if __name__ == "__main__":
    PDF_PATH = f"pdf/1.제품소개_에스피반도체통신.pdf"

    print("PDF 기반 QA 시스템입니다.")
    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            break
        answer = answer_query(PDF_PATH, query)
        print(f"\n🧠 답변: {answer}")
