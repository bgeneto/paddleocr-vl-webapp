# PaddleOCR-VL Document Parser

A production-ready Streamlit application that processes PDF and image files using **PaddleOCR-VL** with a **llama.cpp** backend (Q8 decoder + official mmproj) for state-of-the-art document parsing and OCR. A **vLLM** stack is available via `compose.vllm.yaml`.

## Features

- 📄 **Multi-format Support**: Process PDF, PNG, JPG, JPEG, WEBP, TIFF, and BMP files
- 🔍 **Document Preview**: Preview uploaded PDFs and images before OCR processing
- 🤖 **Advanced Vision-Language Model**: Powered by PaddleOCR-VL-0.9B for accurate text, table, formula, and chart recognition
- 🚀 **Production-Ready**: llama.cpp (default) or vLLM backend for GPU inference
- 📝 **Rich Markdown Output**: Convert complex documents to structured markdown with embedded images
- 👀 **Live Preview**: View processed markdown directly in the browser
- 💾 **Flexible Download**: Download as markdown files or ZIP archives with images
- 🐳 **Docker Compose**: One-command deployment with GPU acceleration
- ⚙️ **Configurable**: Extensive environment-based configuration
- 🔧 **Processing Options**: Document orientation, unwarping, layout detection, and chart recognition

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User's Browser                                │
│                    http://localhost:8501                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                               │
│              (streamlit-app container - Port 8501)                   │
│  • File upload handling                                              │
│  • Processing options UI                                             │
│  • Markdown preview & download                                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ HTTP API
┌─────────────────────────────────────────────────────────────────────┐
│                   PaddleOCR-VL API Service                          │
│             (paddleocr-vl-api container - Port 8080)                │
│  • Layout detection (PP-DocLayoutV2)                                 │
│  • Document preprocessing                                            │
│  • Markdown generation                                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ llama-server API
┌─────────────────────────────────────────────────────────────────────┐
│                   llama.cpp Inference Service                       │
│            (paddleocr-vlm-server container - Port 8080)             │
│  • PaddleOCR-VL-1.6 GGUF (Q8 decoder + official mmproj)             │
│  • GPU-accelerated VLM processing                                    │
│  • Text, table, formula, chart recognition                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware Requirements
- **NVIDIA GPU** + NVIDIA Container Toolkit (layout detection uses GPU on both stacks)
- **GPU VRAM**: llama.cpp default stack typically ~2 GB for the VLM; 8GB+ recommended overall
- **vLLM stack** (`compose.vllm.yaml`): Compute Capability ≥ 8.0, CUDA 12.6+, 8GB+ VRAM (16GB+ preferred)
- **System RAM**: 16GB minimum

### Software Requirements
- Docker >= 19.03
- Docker Compose >= 2.0
- NVIDIA Container Toolkit (nvidia-docker)
- NVIDIA Driver with NVIDIA Container Toolkit (vLLM stack: CUDA 12.6+)

## Quick Start

### 1. Clone and Navigate
```bash
cd streamlit_ocr_app
```

### 2. Create Environment File
```bash
cp .env.example .env
# Edit .env if you need to customize settings
```

### 3. Start the Services
```bash
# Start all services (pulls images and starts containers)
docker compose up -d

# View logs
docker compose logs -f
```

### 4. Access the Application
Open your browser to `http://localhost:8501`

> **Note**: First start downloads ~1.4 GB of GGUF files into `./models/llamacpp` on the host (set `HF_TOKEN` in `.env` for authenticated Hugging Face rate limits). Later starts reuse those files. Check logs with `docker compose logs -f paddleocr-vlm-server`.

### 5. Stop the Services
```bash
docker compose down
```

## Default stack (llama.cpp)

The default stack uses [llama-server](https://github.com/ggml-org/llama.cpp) pinned to `ghcr.io/ggml-org/llama.cpp:server-cuda-b10689`, with [mradermacher/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/mradermacher/PaddleOCR-VL-1.6-GGUF) `PaddleOCR-VL-1.6.Q8_0.gguf` and the official [PaddlePaddle/PaddleOCR-VL-1.6-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF) mmproj (`PaddleOCR-VL-1.6-GGUF-mmproj.gguf`). llama.cpp uses far less VRAM than vLLM (~2 GB for the VLM); the API container still needs a GPU for layout detection while a job is running. After `LLAMA_SLEEP_IDLE_SECONDS` / `PADDLEX_SLEEP_IDLE_SECONDS` of idle time the VLM is unloaded and paddlex is stopped so that memory returns to the GPU.

Throughput is **not** `MAX_PARALLEL_PAGES` / `PAGES_PER_CHUNK` (those stay conservative for vLLM). Use `LLAMA_MAX_PARALLEL_PAGES`, `LLAMA_PAGES_PER_CHUNK`, and `LLAMA_N_PARALLEL` instead.

The paddlex pipeline requires PaddleOCR ≥ 3.5 (`llama-cpp-server` backend). Use a current `API_IMAGE_TAG_SUFFIX` such as `latest-nvidia-gpu-offline`.

## Alternative: vLLM

A second Compose file runs the same Streamlit UI and PaddleOCR-VL API, but serves the VLM with vLLM (or FastDeploy via `VLM_BACKEND`).

Do **not** start this stack while the llama.cpp stack is running — they share `STREAMLIT_HOST_PORT` and typically the same GPU.

```bash
# Stop the default llama.cpp stack first if it is up
docker compose down

docker compose -f compose.vllm.yaml up -d
docker compose -f compose.vllm.yaml logs -f paddleocr-vlm-server
```

First vLLM startup may take 5–10 minutes as the model loads into GPU memory. This stack needs Compute Capability ≥ 8.0 and more VRAM than llama.cpp.

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` and customize:

### Application Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `APP_TITLE` | PaddleOCR-VL Document Parser | Application title |
| `MAX_FILE_SIZE_MB` | 99 | Maximum upload file size |
| `MAX_PDF_PAGES` | 250 | Maximum PDF pages to process |
| `API_TIMEOUT` | 300 | API request timeout in seconds |
| `DATA_CACHE_RETENTION_DAYS` | 45 | Auto-delete cached `data/` entries older than this many days (`0` disables) |
| `APP_UID` | 1000 | Numeric uid for the Streamlit container (`id -u` on the host). Must own `./data` and `./logs`. |
| `APP_GID` | 1000 | Numeric gid for the Streamlit container (`id -g` on the host). |

### Processing Options
| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_QUALITY_FIRST` | false | Master switch. `false` keeps the speed-first pipeline. `true` loads extra detectors, high-recall layout, figure/seal OCR, and cross-page reconstruction. On the llama.cpp stack it also switches KV cache from `q4_0` to `q8_0`. Recreate API, Streamlit, and `paddleocr-vlm-server` after changing it. |
| `USE_DOC_ORIENTATION_CLASSIFY` | false | Auto-detect document orientation (sidebar default; quality-first defaults this on) |
| `USE_DOC_UNWARPING` | false | Correct curved/distorted documents. Stays **off** in quality-first (UVDoc unwarping 500s the llama.cpp VLM). |
| `USE_LAYOUT_DETECTION` | true | Enable layout structure detection |
| `USE_CHART_RECOGNITION` | false | Enable chart/diagram recognition. Stays **off** in quality-first (enable from the sidebar when needed). |
| `PRETTIFY_MARKDOWN` | true | Format markdown for readability |
| `INCLUDE_PAGE_MARKERS` | true | Insert `<!-- Page N -->` comments between pages for RAG chunking (sidebar **Page comments**). Quality-first concatenation is skipped while this is on so page boundaries stay in the markdown. |
| `VISUALIZE_RESULTS` | false | Return processing visualizations |

Set `OCR_QUALITY_FIRST=true` in `.env`, then recreate so the API loads the quality YAML (models are chosen at container start, not per request):

```bash
docker compose up -d --force-recreate
# or: docker compose -f compose.vllm.yaml up -d --force-recreate
```

Speed-first (`false`) is unchanged: same YAML, same request body, same cache keys.

### Docker/Infrastructure
| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_BACKEND` | vllm | Backend for `compose.vllm.yaml` only: `vllm` or `fastdeploy` |
| `GPU_DEVICE_ID` | 0 | GPU device to use |
| `STREAMLIT_HOST_PORT` | 8501 | External port for Streamlit |
| `API_IMAGE_TAG_SUFFIX` | latest-offline | Docker image tag |
| `VLM_IMAGE_TAG_SUFFIX` | latest-offline | VLM image tag (vLLM/FastDeploy stack) |
| `HF_TOKEN` | (empty) | Hugging Face token for faster/authenticated GGUF downloads |
| `LLAMA_CPP_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b10689` | Pinned llama-server CUDA image (do not use floating `:server-cuda`) |
| `LLAMA_HF_REPO` | `mradermacher/PaddleOCR-VL-1.6-GGUF` | Hugging Face repo for the decoder GGUF |
| `LLAMA_HF_FILE` | `PaddleOCR-VL-1.6.Q8_0.gguf` | Decoder filename under `LLAMA_HF_REPO` |
| `LLAMA_MMPROJ_FILE` | `PaddleOCR-VL-1.6-GGUF-mmproj.gguf` | Local mmproj filename in `./models/llamacpp` |
| `LLAMA_MMPROJ_URL` | PaddlePaddle official mmproj | Direct download URL for the vision projector |
| `LLAMA_CTX_SIZE` | 8192 | Desired tokens **per llama-server slot**; total `n_ctx` is `min(CTX_SIZE × N_PARALLEL, CTX_MAX)` |
| `LLAMA_CTX_MAX` | 131072 | Cap on total llama-server `--ctx-size` |
| `LLAMA_MAX_PARALLEL_PAGES` | 2 | Streamlit concurrent API workers for the llama.cpp stack |
| `LLAMA_PAGES_PER_CHUNK` | 22 | Pages per API request on the llama.cpp stack |
| `LLAMA_N_PARALLEL` | 8 | llama-server slots and paddlex VLM `max_concurrency` |
| `LLAMA_UBATCH_SIZE` | 2048 | llama-server `--ubatch-size` (vision prefill). llama.cpp default is 512 |
| `LLAMA_N_GPU_LAYERS` | 99 | Offload all decoder layers to GPU |
| `LLAMA_SLEEP_IDLE_SECONDS` | 60 | Unload the llama-server VLM from VRAM after this many idle seconds (`-1` disables) |
| `PADDLEX_SLEEP_IDLE_SECONDS` | 60 | Stop paddlex after idle so layout-detection VRAM is released (`0` = as soon as the last request finishes, `-1` disables). Recreate `paddleocr-vl-api` after changing. |

## Persistence & Caching

Processed results are persisted to a `data/` directory on the host (bind-mounted into the container at `/app/data`), organized as:

```
data/
└── <sha256-of-file-content>/
    └── <options-fingerprint>/     # e.g. orient=0_unwarp=0_layout=1_chart=0_pretty=1_vis=0_pages=1
                                   # quality-first adds _q=1 so it never reuses a speed cache entry
        ├── meta.json              # display name, hash, options, page count, status, timestamps, sizes
        ├── pages/                 # incremental PDF checkpoints (one JSON file per completed page)
        │   ├── 0000.json
        │   └── 0001.json
        ├── result.md              # extracted markdown (written when the job finishes)
        └── result.zip             # {stem}.md + {stem}_images/... (same layout as the download button)
```

- **Instant re-uploads**: re-uploading a byte-identical file (matched by SHA-256) with the same sidebar options serves the stored result instantly — no API call. The UI shows an **⚡ cached** badge on the file.
- **Container user**: Streamlit runs as `APP_UID`/`APP_GID` (default `1000:1000`, not root). Those values must match the owner of `./data` and `./logs`. Copy them from `id -u` / `id -g` into `.env`, then `docker compose up -d --build --force-recreate streamlit-app`.
- **Changed options**: different sidebar options produce a different fingerprint directory, so the document is reprocessed and the new entry is stored alongside the previous ones. Partial progress from the previous options is not reused.
- **Interrupt / resume**: PDF jobs checkpoint each completed chunk (the pages in one API request) under `pages/`. If OCR is interrupted — vLLM/API crash, timeout, cancel, browser/tab close, or container restart — click **Start OCR** again with the same file and the same options. Already-OCR'd pages are skipped; only the remaining pages are sent to the API. A chunk that was in-flight when the API died is retried as a whole (that request is atomic). Closing the browser does **not** keep the original run going in the background; it only preserves completed pages so the next Start OCR can continue.
- **Survives restarts**: the cache lives on the host bind mount, so `docker compose down && up -d` does not lose it (including partial `pages/` checkpoints).
- **Visualizations**: the Visualizations tab is not available for results restored from the disk cache (only markdown + ZIP are persisted). Reprocess the file to view them.
- **Cleanup**: cache folders older than `DATA_CACHE_RETENTION_DAYS` (default 45) are deleted automatically in a background thread when the Streamlit app runs, so page rendering is not delayed. Age is taken from `meta.json` timestamps when present, otherwise from the folder's last write time. Set the variable to `0` to disable. Manual cleanup still works:

```bash
du -sh data/                              # total cache size
rm -rf data/<sha256>/                     # remove all option variants for one file
rm -rf data/<sha256>/<fingerprint>/       # remove a single options variant
```

The location can be changed with the `DATA_DIR` environment variable (see `.env.example`). Its default is `./data` next to the app (`/app/data` inside Docker, `<repo>/data` for local runs).

## Usage

### Web Interface

1. **Upload Documents**: Drag and drop or click to upload PDF/image files
2. **Preview Document**: Review the uploaded document (first page for PDFs, full image for images)
3. **Configure Options**: Use the sidebar to adjust processing options
4. **Start OCR**: Click the "🚀 Start OCR Processing" button to begin
5. **View Results**: View the extracted markdown in the Preview tab
6. **Download**: Download as `.md` file or `.zip` with embedded images

### API Direct Access

You can also call the PaddleOCR-VL API directly:

```python
import base64
import requests

# Read and encode file
with open("document.pdf", "rb") as f:
    file_data = base64.b64encode(f.read()).decode("ascii")

# Make API request
response = requests.post(
    "http://localhost:8080/layout-parsing",
    json={
        "file": file_data,
        "fileType": 0,  # 0 for PDF, 1 for image
        "useLayoutDetection": True,
        "prettifyMarkdown": True,
    }
)

# Extract markdown
result = response.json()["result"]
for page in result["layoutParsingResults"]:
    print(page["markdown"]["text"])
```

## Troubleshooting

### Services Not Starting
```bash
# Check service status
docker compose ps

# View detailed logs
docker compose logs paddleocr-vlm-server
docker compose logs paddleocr-vl-api
docker compose logs streamlit-app

# vLLM stack (use the matching compose file)
docker compose -f compose.vllm.yaml logs paddleocr-vlm-server
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Check GPU availability in container
docker compose exec paddleocr-vlm-server nvidia-smi
```

### Out of Memory Errors
- Reduce `MAX_FILE_SIZE_MB` and `MAX_PDF_PAGES`
- Process fewer pages at a time
- Use a GPU with more VRAM
- Adjust vLLM memory settings in a custom config

### Slow Processing
- Ensure GPU is being utilized (check `nvidia-smi`)
- Large PDFs may take several minutes
- Consider using FastDeploy backend for specific use cases

## Development

### Local Development (Without Docker)
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit (requires external API service)
PADDLEOCR_VL_API_URL=http://your-api-host:8080/layout-parsing streamlit run app.py
```

> **Note**: For local runs, `DATA_DIR` defaults to `<repo>/data` (gitignored), so the results cache works outside Docker too. Set `DATA_DIR` to use a different location.

### Building Custom Images
```bash
# Build Streamlit frontend
docker build -t paddleocr-vl-streamlit .

# Run with custom API endpoint
docker run -p 8501:8501 \
    -e PADDLEOCR_VL_API_URL=http://your-api:8080/layout-parsing \
    paddleocr-vl-streamlit
```

## Project Structure

```
streamlit_ocr_app/
├── app.py                         # Streamlit application
├── Dockerfile                     # Frontend container definition
├── compose.yaml                   # llama.cpp stack (default; Q8 decoder + official mmproj)
├── compose.vllm.yaml              # vLLM stack (`docker compose -f compose.vllm.yaml`)
├── pipeline_config_llamacpp.yaml  # paddlex pipeline for llama-cpp-server
├── vllm_config.yaml               # vLLM memory/performance config
├── requirements.txt               # Python dependencies
├── .env.example                   # Configuration template
├── .env                           # Local configuration (not in git)
├── .gitignore                     # Git ignore rules
├── .dockerignore                  # Docker build ignore rules
├── data/                          # Processed results cache (gitignored, bind-mounted in Docker)
├── README.md                      # This file
└── logs/                          # Application logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and follows the same Apache 2.0 license.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR toolkit
- [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) - Vision-Language model for document parsing
- [vLLM](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF inference (llama-server)
- [Streamlit](https://streamlit.io/) - Web application framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF preview and processing
