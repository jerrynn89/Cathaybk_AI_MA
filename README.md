# 國泰 AI 商業分析專案

此專案用於將臺積電年度報告 PDF 轉換為 Markdown，並進行 RAG（Retrieval-Augmented Generation）式摘要與問答。主要流程包括：

- `pdftomd.py`: 將 PDF 轉成 Markdown
- `rag_cath.py`: 建立嵌入、FAISS 向量索引、生成摘要、啟動問答系統
## Prototype Demo Video Link
<https://youtu.be/8pvjWL-Q6q0>

## 專案結構

- `pdftomd.py`
  - 使用 `pymupdf4llm` 將 `2024-Annual Report-C.pdf` 轉為 `TSMC_2024.md`
- `rag_cath.py`
  - 將 Markdown 文件切分成多個 chunk
  - 使用本地 `HuggingFaceEmbeddings` 生成向量
  - 建立、載入 FAISS 向量索引
  - 使用 `gpt-4o` 生成摘要與問答
- `TSMC_2024.md`
  - 目前的財報 Markdown 檔案來源
- `artifacts/TSMC_2024/`
  - `summary.md`: 由 `rag_cath.py` 生成的報告摘要
  - `faiss_index/`: FAISS 向量索引儲存位置
  - `rag.log`: 主要執行記錄
- `models/bge-base-zh-v1.5/`
  - 本地中文 embedding 模型目錄

## 主要功能

1. PDF 轉 Markdown
   - 來源檔案：`2024-Annual Report-C.pdf`
   - 目標檔案：`TSMC_2024.md`
2. 生成摘要
   - 將 Markdown 切成段落後，分段進行摘要再合併
   - 產生的摘要存放於 `artifacts/TSMC_2024/summary.md`
3. 問答系統
   - 建置 FAISS 向量索引
   - 從 Markdown 內容中檢索相關 chunk
   - 根據使用者提問提供中文回答

## 使用方式

### 1. 安裝相依套件

建議在虛擬環境中安裝：

```bash
pip install langchain-core langchain-community langchain-huggingface langchain-openai langchain-text-splitters openai pymupdf pymupdf4llm
```

### 2. 轉換 PDF 為 Markdown

```bash
python pdftomd.py
```

### 3. 執行摘要 / 問答

```bash
python rag_cath.py
```

執行後會顯示互動選單：

- `1`: 只生成摘要
- `2`: 只啟動 QA
- `3`: 生成摘要 + 啟動 QA
- `4`: 顯示輸出路徑
- `5`: 離開

### 4. 可用參數

```bash
python rag_cath.py --data_path TSMC_2024.md --embedding_model_path ./models/bge-base-zh-v1.5 --output_dir ./artifacts
```

## 注意事項

- `rag_cath.py` 內部目前會載入 `DEFAULT_DATA_PATH`、`DEFAULT_EMBEDDING_MODEL_PATH` 與 `DEFAULT_OUTPUT_DIR`。
- 若要改成其他報告，請更新 `--data_path`、`--embedding_model_path` 或 `--output_dir`。

## 已完成成果

- `TSMC_2024.md`: 轉換後的臺積電年度報告 Markdown
- `artifacts/TSMC_2024/summary.md`: 由系統生成的中文摘要
- `artifacts/TSMC_2024/faiss_index/`: 問答索引儲存位置
