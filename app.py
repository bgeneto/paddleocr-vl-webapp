"""
PaddleOCR-VL Document Parser Streamlit Application

This application provides a web interface for document OCR using PaddleOCR-VL
with vLLM backend for production-ready inference.
"""

import base64
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF for PDF preview
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

# Configuration from environment variables
APP_TITLE = os.getenv("APP_TITLE", "PaddleOCR-VL Document Parser")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "Upload PDF or image files to convert them to Markdown using PaddleOCR-VL with vLLM backend",
)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "50"))
MAX_PARALLEL_PAGES = int(os.getenv("MAX_PARALLEL_PAGES", "8"))

# PaddleOCR-VL API Configuration
PADDLEOCR_VL_API_URL = os.getenv(
    "PADDLEOCR_VL_API_URL", "http://paddleocr-vl-api:8080/layout-parsing"
)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "300"))

# Processing options
USE_DOC_ORIENTATION_CLASSIFY = (
    os.getenv("USE_DOC_ORIENTATION_CLASSIFY", "false").lower() == "true"
)
USE_DOC_UNWARPING = os.getenv("USE_DOC_UNWARPING", "false").lower() == "true"
USE_LAYOUT_DETECTION = os.getenv("USE_LAYOUT_DETECTION", "true").lower() == "true"
USE_CHART_RECOGNITION = os.getenv("USE_CHART_RECOGNITION", "false").lower() == "true"
PRETTIFY_MARKDOWN = os.getenv("PRETTIFY_MARKDOWN", "true").lower() == "true"
VISUALIZE_RESULTS = os.getenv("VISUALIZE_RESULTS", "false").lower() == "true"

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]


# HTTP Session with connection pooling for better performance
def create_http_session() -> requests.Session:
    """Create an HTTP session with connection pooling and retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )

    # Configure connection pooling - increase pool size for parallel requests
    adapter = HTTPAdapter(
        pool_connections=MAX_PARALLEL_PAGES + 2,
        pool_maxsize=MAX_PARALLEL_PAGES + 2,
        max_retries=retry_strategy,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Global session for connection reuse
_http_session: requests.Session | None = None


def get_http_session() -> requests.Session:
    """Get or create the global HTTP session."""
    global _http_session
    if _http_session is None:
        _http_session = create_http_session()
    return _http_session


def check_api_health() -> bool:
    """Check if the PaddleOCR-VL API service is healthy."""
    try:
        # Extract base URL from the API URL
        base_url = PADDLEOCR_VL_API_URL.rsplit("/", 1)[0]
        health_url = f"{base_url}/health"
        response = requests.get(health_url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def encode_file_to_base64(file_content: bytes) -> str:
    """Encode file content to base64 string."""
    return base64.b64encode(file_content).decode("ascii")


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes."""
    return base64.b64decode(base64_string)


def get_file_type(filename: str) -> int:
    """Determine file type: 0 for PDF, 1 for image."""
    ext = Path(filename).suffix.lower()
    return 0 if ext == ".pdf" else 1


def validate_file(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded file type and size."""
    if uploaded_file is None:
        return False, "No file uploaded"

    file_name = uploaded_file.name.lower()
    file_extension = Path(file_name).suffix

    if file_extension not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {file_extension}"

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return (
            False,
            f"File size ({file_size_mb:.1f}MB) exceeds maximum ({MAX_FILE_SIZE_MB}MB)",
        )

    return True, "Valid"


def get_pdf_preview(file_content: bytes, max_pages: int = 5) -> list[bytes]:
    """Generate preview images from PDF pages."""
    previews = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        num_pages = min(len(doc), max_pages)
        for page_num in range(num_pages):
            page = doc.load_page(page_num)
            # Render at 150 DPI for preview (good balance of quality and size)
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            previews.append(pix.tobytes("png"))
        doc.close()
    except Exception as e:
        st.warning(f"Could not generate PDF preview: {e}")
    return previews


def get_pdf_page_count(file_content: bytes) -> int:
    """Get the number of pages in a PDF."""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


def split_pdf_into_pages(file_content: bytes) -> list[bytes]:
    """Split a PDF into individual single-page PDFs."""
    pages = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(len(doc)):
            # Create a new PDF with just this page
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            page_bytes = single_page_doc.tobytes()
            pages.append(page_bytes)
            single_page_doc.close()
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to split PDF: {e}")
    return pages


def display_file_preview(uploaded_file, file_content: bytes):
    """Display a preview of the uploaded file."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        page_count = get_pdf_page_count(file_content)
        st.caption(f"📑 PDF Document - {page_count} page(s)")

        previews = get_pdf_preview(file_content, max_pages=3)
        if previews:
            cols = st.columns(min(len(previews), 3))
            for idx, preview in enumerate(previews):
                with cols[idx]:
                    st.image(preview, caption=f"Page {idx + 1}", width="stretch")
            if page_count > 3:
                st.caption(f"... and {page_count - 3} more page(s)")
    else:
        # Image preview
        st.caption(f"🖼️ Image - {uploaded_file.size / 1024:.1f} KB")
        st.image(file_content, caption=uploaded_file.name, width="stretch")


def process_document(
    file_content: bytes,
    filename: str,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_layout_detection: bool = True,
    use_chart_recognition: bool = False,
    prettify_markdown: bool = True,
    visualize: bool = False,
) -> dict:
    """
    Process a document using the PaddleOCR-VL API.

    Args:
        file_content: Raw bytes of the file
        filename: Original filename for type detection
        use_doc_orientation_classify: Enable document orientation classification
        use_doc_unwarping: Enable document unwarping
        use_layout_detection: Enable layout detection
        use_chart_recognition: Enable chart recognition
        prettify_markdown: Whether to prettify markdown output
        visualize: Whether to return visualization images

    Returns:
        API response dictionary
    """
    # Encode file to base64
    encoded_file = encode_file_to_base64(file_content)

    # Prepare request payload
    payload = {
        "file": encoded_file,
        "fileType": get_file_type(filename),
        "useDocOrientationClassify": use_doc_orientation_classify,
        "useDocUnwarping": use_doc_unwarping,
        "useLayoutDetection": use_layout_detection,
        "useChartRecognition": use_chart_recognition,
        "prettifyMarkdown": prettify_markdown,
        "visualize": visualize,
    }

    # Make API request using pooled session
    session = get_http_session()
    response = session.post(
        PADDLEOCR_VL_API_URL,
        json=payload,
        timeout=API_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        error_msg = response.json().get("errorMsg", "Unknown error")
        raise RuntimeError(f"API request failed: {error_msg}")

    return response.json()


def process_single_page(args: tuple) -> tuple[int, dict | None, str | None]:
    """
    Process a single PDF page. Helper function for parallel processing.

    Args:
        args: Tuple of (page_num, page_content, filename, options)

    Returns:
        Tuple of (page_num, result_dict, error_message)
    """
    page_num, page_content, filename, options = args
    try:
        page_response = process_document(
            file_content=page_content,
            filename=filename,
            **options,
        )
        result = page_response.get("result", {})
        parsing_results = result.get("layoutParsingResults", [])
        return (page_num, parsing_results, None)
    except Exception as e:
        return (page_num, None, str(e))


def process_pdf_in_batches(
    file_content: bytes,
    filename: str,
    progress_callback=None,
    max_workers: int = None,
    **options,
) -> dict:
    """
    Process a PDF document with parallel page processing for better performance.

    Args:
        file_content: Raw bytes of the PDF file
        filename: Original filename
        progress_callback: Optional callback function(completed_pages, total_pages)
        max_workers: Maximum parallel workers (default: MAX_PARALLEL_PAGES)
        **options: Processing options passed to process_document

    Returns:
        Combined API response dictionary with all pages
    """
    # Split PDF into individual pages
    page_contents = split_pdf_into_pages(file_content)
    total_pages = len(page_contents)

    if total_pages == 0:
        raise RuntimeError("PDF has no pages or could not be read")

    # Use configured max workers or default
    if max_workers is None:
        max_workers = MAX_PARALLEL_PAGES

    # Prepare arguments for parallel processing
    process_args = [
        (page_num, page_content, filename, options)
        for page_num, page_content in enumerate(page_contents)
    ]

    # Process pages in parallel
    results_dict = {}
    errors = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(process_single_page, args): args[0] for args in process_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num, parsing_results, error = future.result()
            completed += 1

            if progress_callback:
                progress_callback(completed, total_pages)

            if error:
                errors.append(f"Page {page_num + 1}: {error}")
            elif parsing_results:
                results_dict[page_num] = parsing_results

    # Report any errors
    if errors:
        error_summary = "; ".join(errors[:5])  # Show first 5 errors
        if len(errors) > 5:
            error_summary += f" ... and {len(errors) - 5} more errors"
        raise RuntimeError(f"Some pages failed to process: {error_summary}")

    # Combine results in page order
    all_parsing_results = []
    for page_num in sorted(results_dict.keys()):
        for pr in results_dict[page_num]:
            all_parsing_results.append(pr)

    # Combine all results into a single response structure
    combined_response = {
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "layoutParsingResults": all_parsing_results,
        },
    }

    return combined_response


def extract_markdown_from_response(api_response: dict) -> tuple[str, dict]:
    """
    Extract markdown text and images from API response.

    Returns:
        Tuple of (markdown_text, images_dict)
    """
    result = api_response.get("result", {})
    parsing_results = result.get("layoutParsingResults", [])

    if not parsing_results:
        return "# No content detected", {}

    markdown_parts = []
    all_images = {}

    for i, page_result in enumerate(parsing_results):
        markdown_info = page_result.get("markdown", {})
        markdown_text = markdown_info.get("text", "")
        images = markdown_info.get("images", {})

        if len(parsing_results) > 1:
            markdown_parts.append(f"<!-- Page {i + 1} -->\n\n{markdown_text}")
        else:
            markdown_parts.append(markdown_text)

        # Prefix image paths with page number for multi-page documents
        for img_path, img_data in images.items():
            if len(parsing_results) > 1:
                all_images[f"page_{i + 1}/{img_path}"] = img_data
            else:
                all_images[img_path] = img_data

    full_markdown = "\n\n---\n\n".join(markdown_parts)
    return full_markdown, all_images


def create_download_zip(markdown_text: str, images: dict, base_filename: str) -> bytes:
    """
    Create a ZIP file containing the markdown and associated images.

    Returns:
        Bytes of the ZIP file
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add markdown file
        zip_file.writestr(f"{base_filename}.md", markdown_text.encode("utf-8"))

        # Add images
        for img_path, img_data in images.items():
            img_bytes = decode_base64_image(img_data)
            zip_file.writestr(f"{base_filename}_images/{img_path}", img_bytes)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def display_processing_options() -> dict:
    """Display and collect processing options from sidebar."""
    st.sidebar.header("⚙️ Processing Options")

    options = {
        "use_doc_orientation_classify": st.sidebar.checkbox(
            "Document Orientation Classification",
            value=USE_DOC_ORIENTATION_CLASSIFY,
            help="Automatically detect and correct document orientation",
        ),
        "use_doc_unwarping": st.sidebar.checkbox(
            "Document Unwarping",
            value=USE_DOC_UNWARPING,
            help="Correct curved or warped document images",
        ),
        "use_layout_detection": st.sidebar.checkbox(
            "Layout Detection",
            value=USE_LAYOUT_DETECTION,
            help="Detect document layout structure (recommended)",
        ),
        "use_chart_recognition": st.sidebar.checkbox(
            "Chart Recognition",
            value=USE_CHART_RECOGNITION,
            help="Enable chart and diagram recognition",
        ),
        "prettify_markdown": st.sidebar.checkbox(
            "Prettify Markdown",
            value=PRETTIFY_MARKDOWN,
            help="Format markdown output for better readability",
        ),
        "visualize": st.sidebar.checkbox(
            "Show Visualization",
            value=VISUALIZE_RESULTS,
            help="Display intermediate processing results (slower)",
        ),
    }

    return options


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"📄 {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)

    # Check API health
    with st.spinner("Checking API service status..."):
        api_healthy = check_api_health()

    if not api_healthy:
        st.error(
            "⚠️ PaddleOCR-VL API service is not available. "
            "Please ensure the service is running."
        )
        st.info(
            f"Expected API URL: `{PADDLEOCR_VL_API_URL}`\n\n"
            "If running locally, start the service with:\n"
            "```bash\ndocker compose up -d\n```"
        )
        return

    st.success("✅ PaddleOCR-VL API service is healthy")

    # Processing options
    options = display_processing_options()

    # File upload section
    st.header("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF or image files",
        type=["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp"],
        accept_multiple_files=True,
        help=f"Supported formats: PDF, PNG, JPG, JPEG, WEBP, TIFF, BMP. Max size: {MAX_FILE_SIZE_MB}MB",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more documents to get started")
        return

    # Initialize session state
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = {}
    if "files_to_process" not in st.session_state:
        st.session_state.files_to_process = {}

    # Store uploaded files content for processing
    valid_files = []
    for uploaded_file in uploaded_files:
        is_valid, validation_msg = validate_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {uploaded_file.name}: {validation_msg}")
            continue

        file_content = uploaded_file.read()
        uploaded_file.seek(0)
        valid_files.append((uploaded_file, file_content))

    if not valid_files:
        return

    # Preview section
    st.header("👁️ Document Preview")
    st.caption("Review your documents before processing")

    for uploaded_file, file_content in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            display_file_preview(uploaded_file, file_content)

            # Store file content for later processing
            st.session_state.files_to_process[file_key] = {
                "name": uploaded_file.name,
                "content": file_content,
            }

    # Start OCR button
    st.header("🚀 Start Processing")

    col1, col2 = st.columns([1, 3])
    with col1:
        start_button = st.button(
            "🔍 Start OCR",
            type="primary",
            width="stretch",
            help="Process all uploaded documents with OCR",
        )
    with col2:
        st.caption(f"Ready to process {len(valid_files)} document(s)")

    if not start_button:
        st.info("👆 Click 'Start OCR' to begin processing your documents")
        return

    # Processing section
    st.header("🔄 Processing Results")

    for uploaded_file, file_content in valid_files:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        is_pdf = uploaded_file.name.lower().endswith(".pdf")

        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            # Check if already processed
            if file_key in st.session_state.processing_results:
                cached = st.session_state.processing_results[file_key]
                st.info("📋 Using cached results. Re-upload to reprocess.")
                markdown_text = cached["markdown"]
                images = cached["images"]
            else:
                try:
                    start_time = time.time()

                    if is_pdf:
                        # Use batch processing for PDFs to handle all pages
                        page_count = get_pdf_page_count(file_content)
                        progress_bar = st.progress(
                            0, text=f"Processing page 1/{page_count}..."
                        )

                        def update_progress(current, total):
                            progress_bar.progress(
                                current / total,
                                text=f"Processing page {current}/{total}...",
                            )

                        api_response = process_pdf_in_batches(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            progress_callback=update_progress,
                            **options,
                        )
                        progress_bar.empty()
                    else:
                        # Process single image directly
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            api_response = process_document(
                                file_content=file_content,
                                filename=uploaded_file.name,
                                **options,
                            )

                    markdown_text, images = extract_markdown_from_response(api_response)

                    processing_time = time.time() - start_time
                    st.success(f"✅ Processed in {processing_time:.1f} seconds")

                    # Cache results
                    st.session_state.processing_results[file_key] = {
                        "markdown": markdown_text,
                        "images": images,
                        "response": api_response,
                    }

                except requests.Timeout:
                    st.error(
                        f"⏱️ Request timed out after {API_TIMEOUT} seconds. "
                        "Try processing smaller files or increase timeout."
                    )
                    continue
                except requests.RequestException as e:
                    st.error(f"🌐 Network error: {str(e)}")
                    continue
                except RuntimeError as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    continue
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    continue

            # Display results in tabs
            tab_preview, tab_raw, tab_download = st.tabs(
                ["📖 Preview", "📝 Raw Markdown", "💾 Download"]
            )

            with tab_preview:
                st.markdown(markdown_text)

                # Display embedded images if any
                if images:
                    st.subheader("🖼️ Extracted Images")
                    cols = st.columns(min(len(images), 3))
                    for idx, (img_path, img_data) in enumerate(images.items()):
                        with cols[idx % 3]:
                            img_bytes = decode_base64_image(img_data)
                            st.image(img_bytes, caption=img_path, width="stretch")

            with tab_raw:
                st.code(markdown_text, language="markdown")

            with tab_download:
                base_filename = Path(uploaded_file.name).stem

                col1, col2 = st.columns(2)

                with col1:
                    # Download markdown only
                    st.download_button(
                        label="📄 Download Markdown (.md)",
                        data=markdown_text.encode("utf-8"),
                        file_name=f"{base_filename}.md",
                        mime="text/markdown",
                    )

                with col2:
                    # Download as ZIP with images
                    if images:
                        zip_data = create_download_zip(
                            markdown_text, images, base_filename
                        )
                        st.download_button(
                            label="📦 Download ZIP (with images)",
                            data=zip_data,
                            file_name=f"{base_filename}_ocr_result.zip",
                            mime="application/zip",
                        )
                    else:
                        st.info("No images to include in ZIP")

            # Display visualization if enabled and available
            if options["visualize"] and file_key in st.session_state.processing_results:
                cached = st.session_state.processing_results[file_key]
                response = cached.get("response", {})
                result = response.get("result", {})
                parsing_results = result.get("layoutParsingResults", [])

                for page_result in parsing_results:
                    output_images = page_result.get("outputImages", {})
                    if output_images:
                        st.subheader("🔍 Processing Visualization")
                        vis_cols = st.columns(min(len(output_images), 2))
                        for idx, (img_name, img_data) in enumerate(
                            output_images.items()
                        ):
                            if img_data:
                                with vis_cols[idx % 2]:
                                    img_bytes = decode_base64_image(img_data)
                                    st.image(
                                        img_bytes,
                                        caption=img_name.replace("_", " ").title(),
                                        width="stretch",
                                    )

    # Batch download section
    if len(st.session_state.processing_results) > 1:
        st.header("📦 Batch Download")
        if st.button("Download All Results as ZIP"):
            batch_zip_buffer = io.BytesIO()
            with zipfile.ZipFile(
                batch_zip_buffer, "w", zipfile.ZIP_DEFLATED
            ) as batch_zip:
                for file_key, data in st.session_state.processing_results.items():
                    base_name = file_key.rsplit("_", 1)[0]  # Remove size suffix
                    base_name = Path(base_name).stem
                    batch_zip.writestr(
                        f"{base_name}/{base_name}.md", data["markdown"].encode("utf-8")
                    )
                    for img_path, img_data in data["images"].items():
                        img_bytes = decode_base64_image(img_data)
                        batch_zip.writestr(f"{base_name}/images/{img_path}", img_bytes)

            batch_zip_buffer.seek(0)
            st.download_button(
                label="📥 Download All Results",
                data=batch_zip_buffer.getvalue(),
                file_name="all_ocr_results.zip",
                mime="application/zip",
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io) and "
        "[PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) using vLLM backend"
    )


if __name__ == "__main__":
    main()
