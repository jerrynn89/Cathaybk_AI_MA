import fitz
from pathlib import Path
import pymupdf4llm


def pdf_to_markdown(pdf_path: str, md_path: str):
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        page_separators=True,
        write_images=False,
        embed_images=False,
        show_progress=True,
    )
    Path(md_path).write_text(md_text, encoding="utf-8")


if __name__ == "__main__":
    input_pdf = "2024-Annual Report-C.pdf"
    output_md = "TSMC_2024.md"

    # 這裡填需要旋轉的頁碼（從0開始）

    pdf_to_markdown(input_pdf, output_md)
    print("完成：已轉成 Markdown")