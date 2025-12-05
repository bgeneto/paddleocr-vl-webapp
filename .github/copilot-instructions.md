# PaddleOCR-VL Document Parser - Copilot Instructions

## Architecture Overview

This is a **three-tier Docker Compose application** for document OCR:

```
Streamlit UI (app.py:8501) → PaddleOCR-VL API (:8080) → vLLM Inference Server
```

- **Frontend**: Single-file Streamlit app (`app.py`) - handles file upload, preview, and results display
- **API Service**: PaddleX container with PP-DocLayoutV2 for layout detection and markdown generation
- **VLM Backend**: vLLM or FastDeploy serving PaddleOCR-VL-0.9B model (GPU-accelerated)

## Key Code Patterns

### Environment-Based Configuration
All settings come from environment variables with defaults. See lines 25-55 in `app.py`:
```python
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
PAGES_PER_CHUNK = int(os.getenv("PAGES_PER_CHUNK", "16"))  # Critical for GPU batching
```

### HTTP Session Pooling
Use `get_http_session()` for API calls - maintains connection pooling with retry logic (see `create_http_session()`).

### PDF Chunked Processing
PDFs are split into multi-page chunks via `split_pdf_into_chunks()` for GPU batching. Key settings:
- `PAGES_PER_CHUNK`: Higher = better GPU utilization (default: 16)
- `MAX_PARALLEL_PAGES`: Concurrent API requests (default: 2)

### Cancellation Pattern
Use `threading.Event` for cooperative cancellation in parallel processing:
```python
cancel_event = threading.Event()
# Check: if cancel_event.is_set(): raise CancellationError(...)
```

### Streamlit Session State Keys
- `st.session_state.processing_results` - Cached OCR results by file key
- `st.session_state.is_processing` - Processing lock
- `st.session_state.cancel_requested` - Cancel flag

## API Integration

The PaddleOCR-VL API expects base64-encoded files:
```python
payload = {
    "file": base64.b64encode(content).decode("ascii"),
    "fileType": 0,  # 0=PDF, 1=image
    "useLayoutDetection": True,
    "prettifyMarkdown": True,
}
response = session.post(PADDLEOCR_VL_API_URL, json=payload, timeout=API_TIMEOUT)
```

Response structure: `result.layoutParsingResults[].markdown.{text, images}`

## Development Commands

```bash
# Start all services (GPU required)
docker compose up -d

# Watch logs during model loading (takes 5-10 min first time)
docker compose logs -f paddleocr-vlm-server

# Local dev (requires running API service)
PADDLEOCR_VL_API_URL=http://localhost:8080/layout-parsing streamlit run app.py

# Rebuild after app.py changes (hot-reload via volume mount)
# No rebuild needed - app.py is mounted read-only
```

## GPU/vLLM Tuning

Edit `vllm_config.yaml` for memory issues:
- `gpu_memory_utilization`: 0.55-0.98 (higher = more batching capacity)
- `max_num_seqs`: Match or exceed `MAX_PARALLEL_PAGES` × `PAGES_PER_CHUNK`
- `enforce_eager: true` if OOM errors persist (disables CUDA graphs)

## File Organization

- `app.py` - **Single source file** for entire Streamlit app (~950 lines)
- `compose.yaml` - Multi-container orchestration with health checks
- `vllm_config.yaml` - vLLM memory/batching configuration
- `requirements.txt` - Minimal deps (streamlit, requests, PyMuPDF, Pillow)

## Conventions

- Use PyMuPDF (`fitz`) for PDF operations, not pdf2image or pdfplumber
- All file validation in `validate_file()` - check size and extension
- Preview generation limited by `MAX_PREVIEW_PAGES` to prevent UI lag
- Image data stored as base64 strings, decoded via `decode_base64_image()`
- Error messages follow emoji prefix pattern: ❌, ⚠️, ✅, etc.
