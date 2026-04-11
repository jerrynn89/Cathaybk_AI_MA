import argparse
import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# =========================
# Default Configuration
# =========================
DEFAULT_DATA_PATH = "TSMC_2024.md"
DEFAULT_EMBEDDING_MODEL_PATH = "./models/bge-base-zh-v1.5"
DEFAULT_OUTPUT_DIR = "./artifacts"

api_key = "your_openai_api_key_here"  # 請替換為你的 OpenAI API Key
MODEL_NAME = "gpt-4o"
CHUNK_SIZE = 450
CHUNK_OVERLAP = 50
RETRIEVE_K = 8
SIMILARITY_THRESHOLD = 1.5
MIN_DOCS = 3
SUMMARY_GROUP_SIZE = 12


SUMMARY_CHUNK_PROMPT = """
You are a financial report analysis assistant.

Please produce a structured summary of the following part of a company's annual report or financial report.

Requirements:
- Output in Chinese
- Use Markdown bullet points
- Write a detailed but concise summary
- Use your own words instead of copying the original text sentence by sentence
- Preserve important facts, business information, financial information, strategic direction, and risk-related content
- If the section contains multiple related points, group them into a coherent summary
- Keep important numbers, percentages, years, plans, or business actions when they are meaningful
- Do not over-compress the content into overly short statements
- Do not use outside knowledge
- Do not simply paste the original wording
- Avoid unnecessary repetition

Please summarize from the following angles whenever applicable:
1. 這段內容主要在說什麼
2. 涉及哪些業務、產品、部門、地區或客群
3. 提到哪些策略、投資、資本支出、發展方向或管理重點
4. 提到哪些財務表現、營運成果、成長/衰退趨勢、重要數字
5. 提到哪些風險、挑戰、不確定性或未來規劃

Output style:
- Use 4–8 bullet points if the content is rich
- Each bullet point should contain enough detail to stand alone
- Prefer “整理後的重點” rather than “極短標題式摘要”

Context:
{context}
"""


SUMMARY_MERGE_PROMPT = """
You are a financial report analysis assistant.

Below are partial summaries of a company's annual report or financial report.
Please merge them into one well-structured Markdown summary.

Requirements:
- Output in Chinese
- Use Markdown format
- Produce a detailed, readable, and well-organized summary
- The summary should be significantly shorter than the original report, but still informative
- Do not make it too brief or overly compressed
- Use your own words to integrate the content
- Merge overlapping points, but retain meaningful details
- Preserve important figures, strategic directions, business developments, risks, and management priorities when relevant
- Do not simply list raw fragments
- Do not repeat similar statements
- Do not use outside knowledge

Recommended structure:
## 一、公司整體概況
- 公司定位、主要業務、整體經營重點

## 二、核心業務與營運重點
- 主要產品/服務/市場/客戶
- 各部門或各區域的營運重點
- 重要營運進展與觀察

## 三、策略與投資方向
- 技術、產能、資本支出、研發、擴張計畫、管理方針
- 中長期發展方向

## 四、財務表現與關鍵數字
- 營收、獲利、成長趨勢、重要財務指標
- 需保留具代表性的數字或趨勢

## 五、風險、挑戰與未來展望
- 市場風險、產業變化、營運挑戰、不確定性
- 公司未來規劃與應對方向

Style guidance:
- Make the summary rich enough that a reader can understand the report without reading the original
- Prefer analytical summary over ultra-short executive bullets
- Each section should contain several meaningful bullet points
- Keep clarity and readability as the top priority

Context:
{context}
"""


QA_PROMPT = """
You are a financial analyst assistant.

Your task is to answer the question based ONLY on the provided context.

Requirements:
- Provide a clear and structured answer
- Use your own words (do NOT copy the original text)
- Summarize and integrate relevant information
- Highlight key facts, numbers, or conclusions when necessary
- Avoid repeating raw sentences from the context
- Avoid listing unprocessed paragraphs
- If multiple data points exist, synthesize them into a meaningful answer
- If the answer is not clearly stated in the context, reply: 不知道
- Answer in Chinese

Context:
{context}

Question:
{question}
"""


# =========================
# Utility Functions
# =========================
def load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_name(path: str) -> str:
    return Path(path).stem.replace(" ", "_")


def get_paths(data_path: str, output_dir: str) -> dict:
    report_name = sanitize_name(data_path)
    report_dir = os.path.join(output_dir, report_name)
    summary_path = os.path.join(report_dir, "summary.md")
    faiss_index_path = os.path.join(report_dir, "faiss_index")
    log_path = os.path.join(report_dir, "rag.log")

    ensure_dir(report_dir)
    return {
        "report_dir": report_dir,
        "summary_path": summary_path,
        "faiss_index_path": faiss_index_path,
        "log_path": log_path,
    }


def log_message(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(message + "\n")


def ask_yes_no(message: str) -> bool:
    while True:
        ans = input(f"{message} (y/n): ").strip().lower()
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("請輸入 y 或 n")


def show_menu():
    print("\n========== 功能選單 ==========")
    print("1. 只生成 Summary")
    print("2. 只啟動 QA")
    print("3. 生成 Summary + 啟動 QA")
    print("4. 顯示輸出路徑")
    print("5. 離開")
    print("================================")


# =========================
# Model / Vector Store
# =========================
def get_embeddings(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到本地 embedding model: {model_path}")

    print(f"正在載入本地 embeddings 模型：{model_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={
            "trust_remote_code": True,
            "device": "cpu",
        },
    )
    print("[OK] 本地 embedding model 載入成功")
    return embeddings


def build_vectorstore(text: str, embeddings) -> FAISS:
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    docs = [Document(page_content=t) for t in chunks]
    return FAISS.from_documents(docs, embeddings)


def get_or_build_vectorstore(
    text: str,
    embeddings,
    faiss_index_path: str,
) -> FAISS:
    if os.path.exists(faiss_index_path):
        print(f"載入本地 FAISS 索引：{faiss_index_path}")
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("[OK] 本地 FAISS 索引載入成功")
        return vectorstore

    print("找不到本地 FAISS 索引，開始建立新索引...")
    vectorstore = build_vectorstore(text, embeddings)
    vectorstore.save_local(faiss_index_path)
    print(f"[OK] FAISS 索引建立完成，已儲存至：{faiss_index_path}")
    return vectorstore


def create_llm():

    if not api_key:
        raise ValueError("請先設定環境變數 OPENAI_API_KEY")

    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )


# =========================
# Summary
# =========================
def summarize_chunk(chunk: str, llm) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_CHUNK_PROMPT)]
    )
    messages = prompt.format_messages(context=chunk)
    response = llm.invoke(messages)
    return response.content.strip() if hasattr(response, "content") else str(response).strip()


def merge_summaries(partial_summaries: List[str], llm) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_MERGE_PROMPT)]
    )
    merged_context = "\n\n".join(partial_summaries)
    messages = prompt.format_messages(context=merged_context)
    response = llm.invoke(messages)
    return response.content.strip() if hasattr(response, "content") else str(response).strip()


def generate_summary_once(text: str, llm, summary_path: str) -> None:
    if os.path.exists(summary_path):
        print(f"[OK] 已找到摘要檔：{summary_path}，略過生成")
        return

    print("開始產生 summary.md ...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    partial_summaries = []
    total_groups = (len(chunks) + SUMMARY_GROUP_SIZE - 1) // SUMMARY_GROUP_SIZE

    for group_idx in range(total_groups):
        start = group_idx * SUMMARY_GROUP_SIZE
        end = min(start + SUMMARY_GROUP_SIZE, len(chunks))
        group_chunks = chunks[start:end]
        group_text = "\n\n".join(group_chunks)

        print(f"摘要中：group {group_idx + 1}/{total_groups}")
        partial_summary = summarize_chunk(group_text, llm)
        partial_summaries.append(partial_summary)

    final_summary = merge_summaries(partial_summaries, llm)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print(f"[OK] 摘要已輸出至：{summary_path}")


def run_summary_pipeline(text: str, llm, summary_path: str, force: bool = False):
    if force and os.path.exists(summary_path):
        os.remove(summary_path)
        print(f"[INFO] 已刪除舊摘要：{summary_path}")

    generate_summary_once(text, llm, summary_path)
    print(f"[INFO] Summary 完成：{summary_path}")


# =========================
# QA
# =========================
def make_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QA_PROMPT),
            ("human", "{question}"),
        ]
    )


def answer_question(
    question: str,
    vectorstore: FAISS,
    llm,
    prompt_template: ChatPromptTemplate
) -> str:
    results = vectorstore.similarity_search_with_score(question, k=RETRIEVE_K)

    if not results:
        return "不知道"

    # 先用 threshold 過濾
    filtered_docs = [doc for doc, score in results if score <= SIMILARITY_THRESHOLD]

    # 如果過濾後太少，至少保留前 MIN_DOCS 個
    if len(filtered_docs) < MIN_DOCS:
        filtered_docs = [doc for doc, _ in results[:MIN_DOCS]]

    if not filtered_docs:
        return "不知道"

    context = "\n\n".join(doc.page_content for doc in filtered_docs)

    messages = prompt_template.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)
    text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    if not text:
        return "不知道"

    lowered = text.lower().strip()
    if lowered in {"不知道", "i don't know", "cannot determine"}:
        # 如果模型仍太保守，再退一步用前更多 chunk 重問一次
        fallback_docs = [doc for doc, _ in results[:5]]
        fallback_context = "\n\n".join(doc.page_content for doc in fallback_docs)

        fallback_messages = prompt_template.format_messages(
            context=fallback_context,
            question=question,
        )
        fallback_response = llm.invoke(fallback_messages)
        fallback_text = (
            fallback_response.content.strip()
            if hasattr(fallback_response, "content")
            else str(fallback_response).strip()
        )

        return fallback_text if fallback_text else "不知道"

    return text


def run_qa_pipeline(
    text: str,
    embedding_model_path: str,
    faiss_index_path: str,
    llm,
    question: str = None,
    rebuild: bool = False,
):
    embeddings = get_embeddings(embedding_model_path)

    if rebuild and os.path.exists(faiss_index_path):
        shutil.rmtree(faiss_index_path)
        print(f"[INFO] 已刪除舊索引：{faiss_index_path}")

    vectorstore = get_or_build_vectorstore(
        text=text,
        embeddings=embeddings,
        faiss_index_path=faiss_index_path,
    )

    prompt_template = make_qa_prompt()

    if question:
        answer = answer_question(question, vectorstore, llm, prompt_template)
        print("\n[ANSWER]")
        print(answer)
        return

    print("已載入 QA 系統。輸入問題後按 Enter（輸入 q 離開）")
    while True:
        q = input("問題: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        print("\n[ANSWER]")
        print(answer_question(q, vectorstore, llm, prompt_template))


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Generic financial report RAG QA")
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="財報 markdown 檔案路徑，例如 TSMC_2024.md",
    )
    parser.add_argument(
        "--embedding_model_path",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_PATH,
        help="本地 embedding model 路徑",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="輸出資料夾，會存放 summary 與 FAISS",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"找不到財報檔案: {args.data_path}")

    paths = get_paths(args.data_path, args.output_dir)
    log_message(paths["log_path"], f"[main] args={args}")

    text = load_md(args.data_path)
    llm = create_llm()

    while True:
        show_menu()
        choice = input("請輸入選項（1/2/3/4/5）: ").strip()

        if choice == "1":
            force_summary = ask_yes_no("是否強制重建 summary？")
            run_summary_pipeline(
                text=text,
                llm=llm,
                summary_path=paths["summary_path"],
                force=force_summary,
            )

        elif choice == "2":
            rebuild_index = ask_yes_no("是否強制重建 FAISS 索引？")


            run_qa_pipeline(
                text=text,
                embedding_model_path=args.embedding_model_path,
                faiss_index_path=paths["faiss_index_path"],
                llm=llm,
                question=None,
                rebuild=rebuild_index,
            )

        elif choice == "3":
            force_summary = ask_yes_no("是否強制重建 summary？")
            rebuild_index = ask_yes_no("是否強制重建 FAISS 索引？")

            run_summary_pipeline(
                text=text,
                llm=llm,
                summary_path=paths["summary_path"],
                force=force_summary,
            )

            run_qa_pipeline(
                text=text,
                embedding_model_path=args.embedding_model_path,
                faiss_index_path=paths["faiss_index_path"],
                llm=llm,
                question=None,
                rebuild=rebuild_index,
            )

        elif choice == "4":
            print("\n========== 輸出路徑 ==========")
            print(f"報告資料夾: {paths['report_dir']}")
            print(f"摘要檔案:   {paths['summary_path']}")
            print(f"FAISS 索引: {paths['faiss_index_path']}")
            print(f"Log 檔案:   {paths['log_path']}")
            print("================================")

        elif choice == "5":
            print("程式結束")
            break

        else:
            print("無效選項，請輸入 1、2、3、4 或 5")


if __name__ == "__main__":
    main()