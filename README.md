# Autot-Web

**Autot-Web** is an **automatic, RAG-based code translation framework** with **multiple context databases (DBs)**, wrapped in a **Gradio** web UI. It lets you:

- pick **any Hugging Face LLM** for generation,
- stream model output **token-by-token** into a large textbox,
- upload source/target **documentation (.txt)** and **transform** them into reusable **vector DBs**,
- manage many context DBs at once, each with a **unique, single-character identifier** (easy to insert/remove in the prompt),
- translate **directories of `.lisp*` files** into modern Common Lisp while preserving functionality,
- view the latest generated **`.autot`** (translated code), **`.comment`** (explanations), and **`.think`** (optional reasoning) files in expandable panels,
- tune **temperature** and **edit the prompt** live.

> The UI file name in this repo: `gradio_lisp_rag_webapp_v2.py`  
> (created from the original request; preserves and extends all functionality you asked for in this chat.)

---

## Table of Contents

- [Features](#features)
- [How it Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the App](#run-the-app)
- [Using the App](#using-the-app)
- [Context DBs & Prompt IDs](#context-dbs--prompt-ids)
- [Outputs](#outputs)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)

---

## Features

- **Hugging Face model picker**
  - Dropdown with common models **plus** a custom “repo id” field (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
- **Streaming output (fixed)**
  - The stream window now shows the **entire cumulative output** as it arrives (not just the last character).
- **Upload & transform context**
  - Separate **Upload** buttons save your **source** and **target** `.txt` docs server-side.
  - **Transform → Vector DB** converts each document set into a JSON vector DB (embeddings + samples).
- **Multiple context DBs**
  - Every transformed DB is auto-registered in a colored list and assigned a **unique, single-character ID**.
  - Insert the selected ID into your prompt at a **visible caret marker** with one click.
  - Because it’s one character, you can **delete it like any normal character**.
- **Consistent prompt expansion for the LLM**
  - Internally, those one-character IDs are expanded into an explicit **mapping section** (with file paths) so the model “knows” which contexts you selected.
- **Directory translation**
  - Point to a folder; the app scans **`**/*.lisp*`** and translates each file.
- **Outputs & history**
  - For each translated file, writes:
    - `*.autot` — translated Lisp
    - `*.comment` — explanations
    - `*.think` — optional hidden reasoning if provided by the model
  - “Latest files” accordions show the most recent files discovered under your input folder.
- **Temperature slider** and **editable prompt**.
- **Best-effort Cancel buttons** next to Load Model, Transform, and Translate.

---

## How it Works

### Architecture

- **Gradio UI**: `gradio_lisp_rag_webapp_v2.py` builds the entire app.
- **Model Manager**: loads a Hugging Face CausalLM + Tokenizer and streams tokens via `TextIteratorStreamer`.
- **RAG Core** (`LispTranslationRAG`):
  - Uses `sentence-transformers` (`all-mpnet-base-v2`) to embed:
    - **Source** docs (what the legacy Lisp code means/does),
    - **Target** docs (the preferred target idioms/implementations),
    - **Done DB** (previously translated results) to keep style/consistency.
  - Extracts **code+context pairs** from docs, plus loose **text chunks**, and stores them in simple JSON “vector DBs”.
  - Builds a **contextual prompt** by combining:
    - Your editable base prompt,
    - Up to a few examples from **source**, **target**, and **previous translations**,
    - The code to translate.
- **Context Registry**
  - Each transformed DB you create is **registered** and assigned a **unique, single-character ID** (e.g., ①, ②, …, Ⓐ, …).
  - When you include those in the prompt, Autot-Web appends a **mapping section** to the model-facing prompt:
    ```
    # Context Databases Selected:
    [CTX ①] -> /absolute/path/to/src_db.json
    [CTX Ⓐ] -> /absolute/path/to/trg_db.json
    ```
  - This keeps the prompt **clean for users** and **unambiguous for the model**.

> Note: cancellation is **best-effort** because the underlying HF generation and large model downloads aren’t always interruptible mid-flight.

---

## Requirements

- Python 3.9+
- Recommended: GPU with recent CUDA for larger models
- Python packages:
  - `gradio`
  - `transformers`
  - `accelerate`
  - `torch` (match your CUDA/CPU environment)
  - `sentence-transformers`
  - `numpy`

---

## Installation

```bash
# 1) Create a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install gradio transformers accelerate sentence-transformers numpy

# 3) Install an appropriate torch build
# GPU example (adjust CUDA version for your system):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Run the App

```bash
python gradio_lisp_rag_webapp_v2.py
```

Gradio will print a local URL (and optionally a public share URL). Open it in your browser.

---

## Using the App

1. **Load a Model**
   - Choose from the dropdown or type a **Hugging Face repo id**, then click **Load Model**.
   - (Optional) Click **Cancel** to request cancellation (best-effort).
2. **Set Temperature & Prompt**
   - Adjust **Temperature** (0.0–1.5).  
   - Edit the **Prompt** text.
3. **Upload Docs & Transform to Vector DBs**
   - Click **Upload Source Docs (.txt)** and **Upload Target Docs (.txt)** to save your doc files server-side.
   - Provide output paths for **Source DB JSON** and **Target DB JSON** (e.g., `src_db.json`, `trg_db.json`).
   - Click **Transform Source → Vector DB** and/or **Transform Target → Vector DB**.
     - Each transformed DB is added to the **Registered Context DBs** palette with a random color and **single-char ID**.
4. **Insert Context IDs into Prompt**
   - Click in the prompt where you want to insert a context ID.
   - Click **Insert caret marker ▮** (places a visible marker).
   - Pick a context ID from the dropdown and click **Insert selected ID into prompt**.  
     The marker ▮ is replaced by the **one-character ID** (easy to remove later).
5. **Pick Input Folder**
   - Set “Directory containing `.lisp*` files”. The app will scan recursively.
6. **Translate**
   - Click **Run Translation on Directory**.  
   - The **Model Output** textbox shows a **live cumulative stream** as tokens arrive.
   - Check the **Latest .autot / .comment / .think** panels to open the most recent generated files.
   - **Cancel** next to Translate is best-effort.
7. **Refresh Latest Files**
   - Click **Refresh latest …** anytime to update the panels from the input folder subtree.

---

## Context DBs & Prompt IDs

- Each vector DB you create is assigned a **unique, single-character** identifier (e.g., ①, ②, Ⓐ).
- These are:
  - **Human-friendly**: visible chips with random colors in the UI.
  - **Prompt-friendly**: a single character you can insert/remove in the prompt like normal text.
- **Internal expansion**: when sending to the LLM, Autot-Web appends a mapping section listing every ID present in your prompt and the file path it refers to. This ensures the model can ground its answers consistently, without cluttering your prompt.

---

## Outputs

For every translated `.lisp*` file:

- `*.autot` — the translated Common Lisp code.
- `*.comment` — an explanation / commentary block.
- `*.think` — if the model produced “hidden” reasoning (e.g., `<think>…</think>`), it’s extracted and saved here.

The UI also maintains:

- `pathlist.txt` — list of processed files.
- `processed_files.txt` — record of which files have already been translated.
- `done_db.json` — a running “previous translations” embedding store used for consistency.

---

## Customization

- **Default models**: `COMMON_MODELS` in the code lists a few popular instruction-tuned models. Add/remove as you like.
- **Prompt building**: edit the default prompt, or customize `_generate_contextual_prompt()` to include more context (e.g., nearest neighbors from vector DBs).
- **Embedding model**: currently `all-mpnet-base-v2`. Change via the `SentenceTransformer` call.
- **File types**: the input glob is `**/*.lisp*`. Adjust if your sources use different extensions.

---

## Troubleshooting

- **Only last character streaming?**  
  Fixed: Autot-Web now feeds the **cumulative** buffer to the textbox every token.
- **Model won’t load / OOM**  
  Use a **smaller** model or ensure a compatible **GPU** with enough VRAM. CPU-only is possible but slow for larger models.
- **Cancellation**  
  “Cancel” is **best-effort** for model loading/transform/translate. HF downloads/generation can’t always be interrupted mid-step.
- **No `.lisp*` files found**  
  Confirm the **Input Folder** path and that files match the pattern.
- **Vector DB empty or small**  
  Ensure your uploaded `.txt` docs have enough content. Autot-Web extracts code/context pairs and text chunks to embed.

---

## FAQ

**Q: Can I use models other than the ones in the dropdown?**  
Yes. Enter any **Hugging Face repo id** (public or available in your environment), then **Load Model**.

**Q: Do the single-character context IDs affect the model?**  
They’re a **user convenience**. For the model, Autot-Web appends a **clear mapping** that resolves each ID to the DB file path so the model can ground its outputs.

**Q: Does Autot-Web retrieve nearest neighbors?**  
The included flow focuses on **prompt priming** with representative samples, plus previous translations. You can extend it with proper ANN search over embeddings to insert top-k neighbors.

**Q: Why Common Lisp specifically?**  
This build targets **translating legacy Lisp code into modern Common Lisp** and preserving behavior, based on source/target docs. You can adapt it to other languages.

---

## License

This project uses open-source libraries with their own licenses (Transformers, Sentence-Transformers, Gradio, etc.). Please review and comply with each library’s license terms when deploying Autot-Web.
