import os
import re
import glob
import json
import hashlib
import threading
import random
import shutil
from typing import Generator, Optional, Tuple, Dict, List

import numpy as np
import gradio as gr

# ------------------------------
# Optional Transformers imports
# ------------------------------
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TextIteratorStreamer,
    )
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

# ------------------------------
# Sentence embeddings
# ------------------------------
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence_transformers is required. Please install: pip install sentence-transformers"
    ) from e


# ==============================================
#                MODEL MANAGER
# ==============================================
class HFModelManager:
    """Load a Hugging Face text-generation model and stream tokens (best‑effort cancel via wrapper)."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.repo_id = None

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, repo_id: str):
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required. Please install: pip install transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cu118 (or appropriate for your system)"
            )
        # Load with device_map auto to use GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            trust_remote_code=True,
        )
        self.repo_id = repo_id

    def generate_stream(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 1024) -> Generator[str, None, None]:
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Load a model first.")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


# ==============================================
#        LISP TRANSLATION RAG (ADAPTED)
# ==============================================
class LispTranslationRAG:
    def __init__(self, src_docs_path, trg_docs_path, model_manager: HFModelManager):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.src_docs_path = src_docs_path
        self.trg_docs_path = trg_docs_path
        self.mm = model_manager

        # Initialize databases with context-aware storage
        self.src_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': []
        }
        self.trg_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': []
        }
        self.done_db = {
            'embeddings': np.zeros((0, 768)),
            'samples': [],  # (code, context) tuples
            'text_embeddings': np.zeros((0, 768)),
            'text_chunks': [],
            'filepaths': []
        }
        self.translation_cache = {}

    # --- Persistence for context DBs --------------------------------------

    def _save_db(self, db, path):
        try:
            data = {
                'embeddings': db['embeddings'].tolist() if isinstance(db['embeddings'], np.ndarray) else db['embeddings'],
                'samples': db['samples'],
                'text_embeddings': db['text_embeddings'].tolist() if isinstance(db['text_embeddings'], np.ndarray) else db['text_embeddings'],
                'text_chunks': db['text_chunks'],
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save DB to {path} - {str(e)}")

    def _load_db(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            db = {
                'embeddings': np.array(data.get('embeddings', [])),
                'samples': data.get('samples', []),
                'text_embeddings': np.array(data.get('text_embeddings', [])),
                'text_chunks': data.get('text_chunks', []),
            }
            if db['embeddings'].size == 0:
                db['embeddings'] = np.zeros((0, 768))
            if db['text_embeddings'].size == 0:
                db['text_embeddings'] = np.zeros((0, 768))
            return db
        except Exception as e:
            print(f"Warning: Could not load DB from {path} - {str(e)}")
            return None

    def prepare_context_dbs(self, src_db_path, trg_db_path):
        # Source DB
        if src_db_path and os.path.exists(src_db_path):
            loaded = self._load_db(src_db_path)
            if loaded:
                self.src_db = loaded
                print(f"Loaded source context DB from: {src_db_path}")
            else:
                print(f"Rebuilding source DB from docs due to load failure.")
                self._build_enhanced_database(self.src_docs_path, self.src_db)
                self._save_db(self.src_db, src_db_path)
        else:
            print(f"Building source context DB from docs...")
            self._build_enhanced_database(self.src_docs_path, self.src_db)
            if src_db_path:
                self._save_db(self.src_db, src_db_path)

        # Target DB
        if trg_db_path and os.path.exists(trg_db_path):
            loaded = self._load_db(trg_db_path)
            if loaded:
                self.trg_db = loaded
                print(f"Loaded target context DB from: {trg_db_path}")
            else:
                print(f"Rebuilding target DB from docs due to load failure.")
                self._build_enhanced_database(self.trg_docs_path, self.trg_db)
                self._save_db(self.trg_db, trg_db_path)
        else:
            print(f"Building target context DB from docs...")
            self._build_enhanced_database(self.trg_docs_path, self.trg_db)
            if trg_db_path:
                self._save_db(self.trg_db, trg_db_path)

    # --- Doc processing ----------------------------------------------------

    def _extract_code_context_pairs(self, text):
        sections = re.split(r'\n\s*\n', text)
        pairs = []
        for section in sections:
            code_blocks = re.findall(r'(?:^|\n)(?:;+\s*Example:?\s*)?(\(.*?\))(?=\n|$)', section, re.DOTALL)
            if code_blocks:
                context = re.sub(r'(\(.*?\))', '', section)
                context = ' '.join(context.split()).strip()
                for code in code_blocks:
                    if code.strip():
                        pairs.append((code.strip(), context))
        return pairs

    def _process_doc_file(self, filepath):
        if not filepath or not os.path.exists(filepath):
            print(f"Error: Documentation file {filepath} not found")
            return [], []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        code_context_pairs = self._extract_code_context_pairs(content)
        text_chunks = [
            chunk.strip() for chunk in
            re.split(r'\n\s*\n', re.sub(r'\(.*?\)', '', content))
            if chunk.strip() and len(chunk.split()) > 5
        ]
        return code_context_pairs, text_chunks

    def _build_enhanced_database(self, doc_path, db):
        code_context_pairs, text_chunks = self._process_doc_file(doc_path)
        for code, context in code_context_pairs:
            try:
                code_embedding = self.model.encode(code).reshape(1, -1)
                context_embedding = self.model.encode(context).reshape(1, -1)
                db['embeddings'] = code_embedding if db['embeddings'].shape[0] == 0 else np.vstack([db['embeddings'], code_embedding])
                db['samples'].append((code, context))
                db['text_embeddings'] = context_embedding if db['text_embeddings'].shape[0] == 0 else np.vstack([db['text_embeddings'], context_embedding])
                db['text_chunks'].append(context)
            except Exception as e:
                print(f"Error processing sample: {str(e)}")
        for chunk in text_chunks:
            try:
                chunk_embedding = self.model.encode(chunk).reshape(1, -1)
                db['text_embeddings'] = chunk_embedding if db['text_embeddings'].shape[0] == 0 else np.vstack([db['text_embeddings'], chunk_embedding])
                db['text_chunks'].append(chunk)
            except Exception as e:
                print(f"Error processing text chunk: {str(e)}")

    # --- Done DB persistence ----------------------------------------------

    def _load_done_db(self):
        if os.path.exists('done_db.json'):
            try:
                with open('done_db.json', 'r') as f:
                    data = json.load(f)
                    self.done_db['embeddings'] = np.array(data.get('embeddings', []))
                    self.done_db['samples'] = data.get('samples', [])
                    self.done_db['filepaths'] = data.get('filepaths', [])
                    if self.done_db['embeddings'].size == 0:
                        self.done_db['embeddings'] = np.zeros((0, 768))
            except Exception as e:
                print(f"Warning: Could not load done_db - {str(e)}")

    def _save_done_db(self):
        try:
            with open('done_db.json', 'w') as f:
                json.dump({
                    'embeddings': self.done_db['embeddings'].tolist(),
                    'samples': self.done_db['samples'],
                    'filepaths': self.done_db['filepaths']
                }, f)
        except Exception as e:
            print(f"Warning: Could not save done_db - {str(e)}")

    def _update_done_db(self, filepath, source_code):
        try:
            processed_code = self._preprocess_code(source_code)
            embedding = self.model.encode(processed_code).reshape(1, -1)
            self.done_db['embeddings'] = embedding if self.done_db['embeddings'].shape[0] == 0 else np.vstack([self.done_db['embeddings'], embedding])
            self.done_db['samples'].append(processed_code)
            self.done_db['filepaths'].append(filepath)
            self._save_done_db()
        except Exception as e:
            print(f"Warning: Could not update done_db - {str(e)}")

    # --- Utilities ---------------------------------------------------------

    def _preprocess_code(self, code):
        code = re.sub(r';.*', '', code)
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def _extract_code_block(self, text):
        match = re.search(r'```lisp\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _extract_comments_block(self, text):
        match = re.search(r'```comments\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "No explanations provided"

    def _extract_think_block(self, text):
        m1 = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        if m1:
            return m1.group(1).strip()
        m2 = re.search(r'```think\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
        return None

    def _write_output(self, path, content):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    # --- Prompt builder ----------------------------------------------------

    def _generate_contextual_prompt(self, source_code, user_prompt_override: Optional[str] = None):
        if user_prompt_override and user_prompt_override.strip():
            base = user_prompt_override.strip()
        else:
            base = (
                "Translate this Lisp code to modern Common Lisp while preserving all functionality.\n"
                "The first knowledge source provided shall help you understand what this lisp code actually does, the second language source describes the target implementation, so adhere to it in your answers. The third knowledge source represents what you have done so far: You must always remain consistent to it!\n"
                "While translating this Common Lisp code, always proceed step by step: What is the expected input? What does the source code you shall translate do? How do you preserve all its functionality using the target implementation?\n"
                "The Lisp code you shall translate into modern Common Lisp, while preserving all it's functionality is as follows:"
            )
        prompt_parts = [base]

        if self.src_db['samples']:
            prompt_parts.append("\nSource Examples:")
            for code, ctx in self.src_db['samples'][:3]:
                prompt_parts.append(f"\nContext: {ctx}\nCode: {code}")

        if self.trg_db['samples']:
            prompt_parts.append("\nTarget Examples:")
            for code, ctx in self.trg_db['samples'][:3]:
                prompt_parts.append(f"\nContext: {ctx}\nCode: {code}")

        if self.done_db['samples']:
            prompt_parts.append("\nPrevious Translations:")
            for sample in self.done_db['samples'][-3:]:
                prompt_parts.append(f"\n{sample}")

        prompt_parts.append(f"\nCode to translate:\n{source_code}")
        prompt_parts.append("\nProvide the translated code in a ```lisp block and explanations in a ```comments block. If you include chain-of-thought or hidden reasoning, wrap it in <think>...</think> (or a ```think block).")

        return '\n'.join(prompt_parts)

    # --- Translation methods ----------------------------------------------

    def translate_file_stream(self, input_path: str, temperature: float = 0.7, user_prompt_override: Optional[str] = None) -> Generator[str, None, Tuple[str, str, Optional[str]]]:
        """Stream model output while writing .autot/.comment/.think when finished.
        Yields cumulative text chunks; returns a tuple (translated_code, comments, think) when done.
        """
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            file_hash = hashlib.md5(source_code.encode()).hexdigest()
            if file_hash in self.translation_cache:
                translated_code, comments, think = self.translation_cache[file_hash]
                base_path = os.path.splitext(input_path)[0]
                self._write_output(f"{base_path}.autot", translated_code)
                self._write_output(f"{base_path}.comment", comments)
                if think:
                    self._write_output(f"{base_path}.think", think)
                yield "(loaded from cache)\n"
                return translated_code, comments, think

            prompt = self._generate_contextual_prompt(source_code, user_prompt_override)

            full_output = ""
            for token in self.mm.generate_stream(prompt=prompt, temperature=temperature):
                full_output += token
                # IMPORTANT: yield cumulative text so the UI shows the whole stream so far
                yield full_output

            translated_code = self._extract_code_block(full_output)
            comments = self._extract_comments_block(full_output)
            think = self._extract_think_block(full_output)

            base_path = os.path.splitext(input_path)[0]
            self._write_output(f"{base_path}.autot", translated_code)
            self._write_output(f"{base_path}.comment", comments)
            if think:
                self._write_output(f"{base_path}.think", think)

            self._update_done_db(input_path, source_code)
            self.translation_cache[file_hash] = (translated_code, comments, think)
            return translated_code, comments, think
        except Exception as e:
            yield f"\n[Error] Failed to translate {input_path}: {str(e)}\n"
            return None, str(e), None

    def translate_directory_stream(self, input_dir: str, temperature: float = 0.7, user_prompt_override: Optional[str] = None) -> Generator[str, None, None]:
        if not os.path.exists(input_dir):
            yield f"Error: Input directory {input_dir} does not exist\n"
            return
        all_files = []
        for filepath in glob.glob(os.path.join(input_dir, '**/*.lisp*'), recursive=True):
            all_files.append(filepath)
        if not all_files:
            yield "No .lisp* files found in the selected directory.\n"
            return

        with open("pathlist.txt", "w") as f1:
            for filepath in all_files:
                f1.write(f"{filepath}\n")

        processed_files = set()
        if os.path.exists("processed_files.txt"):
            with open("processed_files.txt", "r") as f3:
                processed_files = {line.strip() for line in f3 if line.strip()}

        with open("processed_files.txt", "a") as f3:
            for path in all_files:
                if path not in processed_files:
                    prefix = f"\n=== Translating: {path} ===\n"
                    so_far = prefix
                    yield so_far
                    for chunk in self.translate_file_stream(path, temperature=temperature, user_prompt_override=user_prompt_override):
                        so_far = chunk
                        yield so_far
                    f3.write(f"{path}\n")
                    f3.flush()
                    processed_files.add(path)
                    so_far += "\n--- Done ---\n"
                    yield so_far


# ==============================================
#           CONTEXT REGISTRY & HELPERS
# ==============================================
ctx_registry: Dict[str, Dict[str, str]] = {}
used_symbols: List[str] = []
UNIQUE_SYMBOLS = list("①②③④⑤⑥⑦⑧⑨⑩⓪⓵⓶⓷⓸⓹⓺⓻⓼⓽ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ❶❷❸❹❺❻❼❽❾❿")

def random_color() -> str:
    return f"hsl({random.randint(0,359)}, 70%, 45%)"

def register_ctx_db(path: str) -> Dict[str, str]:
    # assign unique single-character symbol so user can delete it as one character
    sym = None
    for s in UNIQUE_SYMBOLS:
        if s not in used_symbols:
            sym = s
            break
    if sym is None:
        sym = chr(random.randint(0x2460, 0x24FF))
    used_symbols.append(sym)
    color = random_color()
    ctx_registry[sym] = {"path": path, "color": color}
    return {"symbol": sym, "color": color, "path": path}

def build_ctx_html() -> str:
    parts = ["<div style='display:flex;gap:8px;flex-wrap:wrap'>"]
    for sym, meta in ctx_registry.items():
        parts.append(
            f"<span style='padding:4px 8px;border-radius:12px;background:{meta['color']};color:white;font-weight:600'> {sym} </span>"
        )
    parts.append("</div>")
    return "".join(parts)

def expand_prompt_with_ctx(prompt_text: str) -> str:
    # Add machine-readable context mapping at the end, using unique one-char IDs present in the prompt
    ctx_seen = [sym for sym in ctx_registry.keys() if sym in prompt_text]
    if not ctx_seen:
        return prompt_text
    ctx_section = ["\n\n# Context Databases Selected:"]
    for sym in ctx_seen:
        ctx_section.append(f"[CTX {sym}] -> {ctx_registry[sym]['path']}")
    return prompt_text + "\n" + "\n".join(ctx_section)

def insert_ctx_symbol(current_prompt: str, symbol: str, caret_marker: str = "▮"):
    # Insert at a visible caret marker ▮ if present, else append
    if caret_marker and caret_marker in current_prompt:
        return current_prompt.replace(caret_marker, symbol, 1)
    return current_prompt + symbol

# ==============================================
#                  GRADIO UI
# ==============================================
mm = HFModelManager()
translator = LispTranslationRAG(src_docs_path='./src_docs.txt', trg_docs_path='./trg_docs_2.txt', model_manager=mm)

DEFAULT_PROMPT = (
    "Translate this Lisp code to modern Common Lisp while preserving all functionality.\n"
    "(You can edit this prompt before running.)"
)

COMMON_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Upload helpers

def save_uploaded(file_obj, dest_name):
    if file_obj is None:
        return "No file uploaded.", None
    dest = os.path.abspath(dest_name)
    shutil.copy(file_obj.name, dest)
    return f"Saved to {dest}", dest

# Transform helpers (best-effort, runs inline; users can click Cancel status message only)

def transform_src(out_path):
    translator._build_enhanced_database(translator.src_docs_path, translator.src_db)
    if out_path:
        translator._save_db(translator.src_db, out_path)
        register_ctx_db(out_path)
    return f"Source context DB saved to {out_path}", build_ctx_html(), list(ctx_registry.keys())

def transform_trg(out_path):
    translator._build_enhanced_database(translator.trg_docs_path, translator.trg_db)
    if out_path:
        translator._save_db(translator.trg_db, out_path)
        register_ctx_db(out_path)
    return f"Target context DB saved to {out_path}", build_ctx_html(), list(ctx_registry.keys())

# Streaming wrapper

def stream_directory(input_dir, temperature, user_prompt):
    if not mm.is_ready():
        yield "Please load a model first.\n"
        return
    expanded_prompt = expand_prompt_with_ctx(user_prompt)
    so_far = ""
    for chunk in translator.translate_directory_stream(input_dir=input_dir, temperature=temperature, user_prompt_override=expanded_prompt):
        so_far = chunk
        yield so_far


def do_load_model(preset_repo: str, custom_repo: str):
    repo = custom_repo.strip() if custom_repo and custom_repo.strip() else preset_repo
    if not repo:
        return gr.update(value=""), f"Please select or enter a Hugging Face model id.", False
    try:
        mm.load(repo)
        return gr.update(value=repo), f"Loaded model: {repo}", True
    except Exception as e:
        return gr.update(value=preset_repo), f"Failed to load model: {e}", False


def find_latest_files(in_dir: str):
    def latest_with_ext(ext: str) -> Optional[str]:
        files = glob.glob(os.path.join(in_dir, f"**/*{ext}"), recursive=True)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    latest_autot = latest_with_ext('.autot')
    latest_comment = latest_with_ext('.comment')
    latest_think = latest_with_ext('.think')

    def safe_read(p: Optional[str]) -> str:
        if p and os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                return f"[Could not read {p}: {e}]"
        return "[No file found]"

    return (
        latest_autot or "",
        safe_read(latest_autot),
        latest_comment or "",
        safe_read(latest_comment),
        latest_think or "",
        safe_read(latest_think),
    )


def build_ui():
    with gr.Blocks(title="Lisp Translation RAG – Hugging Face Edition") as demo:
        gr.Markdown("# Lisp Translation RAG – Gradio Webapp (Hugging Face models)")

        with gr.Row():
            model_dropdown = gr.Dropdown(choices=COMMON_MODELS, value=COMMON_MODELS[0], label="Select HF model")
            custom_model = gr.Textbox(label="...or enter custom repo id", placeholder="e.g. meta-llama/Llama-3.1-8B-Instruct")
            loaded_model = gr.Textbox(label="Loaded Model", interactive=False)
        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            cancel_load_btn = gr.Button("Cancel")
        load_status = gr.Markdown(visible=True)

        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            user_prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt (editable)", lines=4)
        gr.Markdown("Click in the prompt, then press **Insert caret marker ▮**. The **Insert** button will replace that marker with a colored context ID (single character).")
        caret_btn = gr.Button("Insert caret marker ▮")

        gr.Markdown("## Context Docs & DBs")
        with gr.Row():
            upload_src = gr.UploadButton("Upload Source Docs (.txt)", file_types=[".txt"], file_count="single")
            save_src_status = gr.Markdown()
            upload_trg = gr.UploadButton("Upload Target Docs (.txt)", file_types=[".txt"], file_count="single")
            save_trg_status = gr.Markdown()
        with gr.Row():
            src_db_json = gr.Textbox(value="src_db.json", label="Source DB JSON path (save/load)")
            trg_db_json = gr.Textbox(value="trg_db.json", label="Target DB JSON path (save/load)")
        with gr.Row():
            transform_src_btn = gr.Button("Transform Source → Vector DB")
            cancel_transform_src_btn = gr.Button("Cancel")
            transform_trg_btn = gr.Button("Transform Target → Vector DB")
            cancel_transform_trg_btn = gr.Button("Cancel")
        ctx_status = gr.Markdown()

        gr.Markdown("### Registered Context DBs")
        ctx_html = gr.HTML(value=build_ctx_html())
        ctx_choice = gr.Dropdown(choices=list(ctx_registry.keys()), label="Select context ID to insert")
        insert_btn = gr.Button("Insert selected ID into prompt")

        gr.Markdown("## Input Folder")
        input_dir = gr.Textbox(value="./symbolics/sys.sct", label="Directory containing .lisp* files")

        with gr.Row():
            run_btn = gr.Button("Run Translation on Directory", variant="primary")
            cancel_translate_btn = gr.Button("Cancel")
            refresh_btn = gr.Button("Refresh latest .autot/.comment/.think")

        gr.Markdown("## Streaming LLM Output")
        stream_box = gr.Textbox(label="Model Output (streaming)", lines=18)

        gr.Markdown("## Latest generated files (expand to view)")
        with gr.Accordion("Latest .autot", open=False):
            latest_autot_path = gr.Textbox(label="Path", interactive=False)
            latest_autot_text = gr.Textbox(label="Contents", lines=10)
        with gr.Accordion("Latest .comment", open=False):
            latest_comment_path = gr.Textbox(label="Path", interactive=False)
            latest_comment_text = gr.Textbox(label="Contents", lines=10)
        with gr.Accordion("Latest .think", open=False):
            latest_think_path = gr.Textbox(label="Path", interactive=False)
            latest_think_text = gr.Textbox(label="Contents", lines=10)

        # --- Wiring ---
        load_btn.click(
            do_load_model,
            inputs=[model_dropdown, custom_model],
            outputs=[loaded_model, load_status, run_btn],
        )
        cancel_load_btn.click(lambda: "Cancel requested (best effort; model loading may continue).", outputs=[load_status])

        # caret marker insertion
        caret_btn.click(lambda txt: (txt + ("" if txt.endswith("\n") else "") + "▮"), inputs=[user_prompt], outputs=[user_prompt])

        # Upload saves
        upload_src.upload(fn=lambda f: save_uploaded(f, "uploaded_src_docs.txt"), inputs=[upload_src], outputs=[save_src_status, gr.State()])
        upload_trg.upload(fn=lambda f: save_uploaded(f, "uploaded_trg_docs.txt"), inputs=[upload_trg], outputs=[save_trg_status, gr.State()])

        # Transform buttons
        transform_src_btn.click(transform_src, inputs=[src_db_json], outputs=[ctx_status, ctx_html, ctx_choice])
        cancel_transform_src_btn.click(lambda: "Cancel requested (best effort).", outputs=[ctx_status])
        transform_trg_btn.click(transform_trg, inputs=[trg_db_json], outputs=[ctx_status, ctx_html, ctx_choice])
        cancel_transform_trg_btn.click(lambda: "Cancel requested (best effort).", outputs=[ctx_status])

        # Stream output to textbox
        run_btn.click(
            stream_directory,
            inputs=[input_dir, temperature, user_prompt],
            outputs=[stream_box],
        )
        cancel_translate_btn.click(lambda: "(Stop the cell/run to hard-cancel; this button is best-effort)", outputs=[])

        # Refresh latest generated files
        refresh_btn.click(
            find_latest_files,
            inputs=[input_dir],
            outputs=[
                latest_autot_path, latest_autot_text,
                latest_comment_path, latest_comment_text,
                latest_think_path, latest_think_text,
            ],
        )

        # Insert selected context ID into prompt at caret marker
        insert_btn.click(lambda prompt_text, sym: insert_ctx_symbol(prompt_text, sym), inputs=[user_prompt, ctx_choice], outputs=[user_prompt])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
