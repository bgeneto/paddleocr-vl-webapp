# Persistence Volume + Hash-Based Result Cache

## Goal

1. Persist OCR outputs (markdown + ZIP with images) to a durable volume so they survive container recreation.
2. When a user re-uploads a byte-identical file (SHA-256) with the same processing options, serve the stored md/zip instantly — no API call.

## Resolved decisions

| Decision | Choice |
|---|---|
| Volume type | **Bind mount** `./data` → `/app/data` (writable by `appuser`). Remove the dead, never-mounted named volume `uploads` from `compose.yaml:126-128`. Seed with committed `data/.gitkeep` so the host dir is created with the invoking user's ownership (avoid root-owned bind mount breaking non-root `appuser`). |
| Cache key | `data/<sha256-of-content>/<options-fingerprint>/` — fingerprint = deterministic compact string of the 6 sidebar booleans (e.g. `orient=1_unwarp=0_layout=1_chart=0_pretty=1_vis=0`). Same file + different options → separate subdir, never a stale hit. |
| Artifacts stored per entry | `meta.json` (display name, hash, fingerprint, options, page count, timestamps/sizes), `result.md`, `result.zip`. Internal names fixed; the user-facing download names come from `meta.json.display_name`. |
| Visualization tab on cache hit | **Not restored** (API `outputImages` are not persisted — user asked for md+zip only). Show a notice in that tab when the result came from cache. Accepted limitation, documented in README. |
| Pruning/eviction | **Out of scope.** Manual cleanup of `./data` (documented). |

## Target data layout

```
./data/                                  # host bind mount → /app/data
  <sha256hex>/
    <options-fingerprint>/
      meta.json
      result.md
      result.zip                          # contains {stem}.md + {stem}_images/... exactly like create_download_zip() today
```

`DATA_DIR` env var (new, in `.env.example`): default = `Path(__file__).parent / "data"` → resolves to `/app/data` in the container and `<repo>/data` for local dev; overridable.

## Changes (ordered)

### 1. `compose.yaml`
- `streamlit-app.volumes`: add `- ./data:/app/data` (rw). Keep `./app.py:...:ro` and `./logs`.
- Add `- DATA_DIR=${DATA_DIR:-/app/data}` to `streamlit-app.environment`.
- Delete the top-level `volumes: uploads:` block (unused).

### 2. `Dockerfile`
- In the existing `RUN mkdir -p ... chown` step (Dockerfile:39-40): add `/app/data` to the mkdir/chown list so the image-side dir exists with `appuser` ownership (also correct if the bind mount is absent).

### 3. Repo housekeeping
- Add `data/.gitkeep` (commit it).
- `.gitignore`: add `data/` (existing `*.zip` rule already covers contents, but ignore the dir explicitly).
- `.dockerignore`: add `data/` so host data is never baked into the image.
- `.env.example`: add `DATA_DIR=/app/data` under a new "Storage / Caching" section with a short comment.

### 4. `app.py` — new helpers (module level, near existing config)

- `DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data")))`; `DATA_DIR.mkdir(parents=True, exist_ok=True)` at import/startup.
- `compute_sha256(content: bytes) -> str` — `hashlib.sha256` (content already fully in memory; 99 MB hashes in well under a second).
- `options_fingerprint(options: dict) -> str` — canonical, deterministic string of the 6 booleans in fixed key order (see layout example above).
- `cache_dirs(hash_str, fingerprint) -> Path` — `DATA_DIR / hash_str / fingerprint`, `mkdir(parents=True, exist_ok=True)`.
- `load_cached_result(hash_str, fingerprint) -> dict | None` — returns `None` on any miss/error. On hit: read `meta.json`, `result.md`; open `result.zip` (read mode), extract every member under `{display_stem}_images/` as raw bytes and re-encode to base64 to reconstruct the existing `images` dict shape (keys `page_N/filename` or `filename`, matching `extract_markdown_from_response`) so all downstream display/download code is untouched. On `BadZipFile`/missing/corrupt entry → log warning, return `None` (treated as miss; do NOT auto-delete).
- `save_result_to_disk(hash_str, fingerprint, display_name, markdown_text, images, page_count) -> None` — wraps in try/except (a storage failure must never break a successful in-memory result; log + `st.warning`). Builds `result.zip` via existing `create_download_zip()` (pass the sanitized stem as `base_filename`), then writes `result.md`, `result.zip`, `meta.json` each **atomically**: write to `<name>.tmp` in the same dir, `os.replace()` to final name. `display_stem = Path(display_name).stem.strip() or "document"`.

### 5. `app.py` — main-loop integration (`main()`, per-file block around app.py:742-832)

For each valid uploaded file, in this order:

1. Compute `file_hash = compute_sha256(file_content)` and `fingerprint = options_fingerprint(options)` right after `validate_file` (store both in the session file dict alongside name/content).
2. **Session cache** (existing check, app.py:754) → keep as-is, fastest path.
3. **Disk cache**: if not in session cache, call `load_cached_result(...)`. On hit: populate `st.session_state.processing_results[file_key]` identically to the fresh-processing path (markdown, images) so the Preview / Raw / Download / batch-download code all work unchanged; show `st.success("⚡ Instant from disk cache — file previously processed with these options")` instead of the progress bar/API call.
4. **Fresh processing** (existing API flow). On success, in addition to the existing session-state store (app.py:808-812), call `save_result_to_disk(...)` (skip if save raises — already handled inside the helper).
5. Visualization block (app.py:913-934): when the result originated from disk cache, skip it and show a caption noting visualizations aren't stored in the cache.

Update the existing "Using cached results" info text (app.py:756) to distinguish session vs disk origin.

### 6. `README.md`
- New section "Persistence & Caching": explains `./data` layout, that re-uploading an identical file with identical options is instant, that changing sidebar options reprocesses, that the Visualizations tab is unavailable for cached results, and manual cleanup (`rm -rf data/*/...` / `du -sh data`).
- Project Structure tree (README:228-241): add `data/` entry.
- Dev section (README:206-213): note `DATA_DIR` defaults to `<repo>/data` for local runs.

## Failure modes & edge cases

- **Corrupt/partial zip on disk** → `load_cached_result` returns None → normal reprocess; stale entry left in place for inspection.
- **Disk full / permission error on save** → caught inside `save_result_to_disk`; user still gets working in-memory results + warnings.
- **Concurrent identical uploads** (two browsers) → both process, both write atomically; last writer wins; no corruption.
- **Same content, renamed file** → same hash dir; `display_name` in meta drives download names per original upload.
- **Options change** → different fingerprint dir → reprocess; old entries coexist (documented; manual cleanup).
- **Container restart** → session cache empty, disk cache intact → still instant.
- **Non-docker local dev** → `DATA_DIR` default resolves to `<repo>/data`, gitignored.

## Validation

1. **Unit (no GPU/API needed):** local Python snippet importing `app.py`'s helpers: sha256 of a known string; fingerprint determinism/order-independence; `save_result_to_disk` → `load_cached_result` round-trip (markdown equal, images dict keys/values equal, zip member list matches); corrupt-zip case returns None.
2. **E2E (docker compose up -d):**
   - Upload small PDF #1 → processes slowly; verify `data/<hash>/<fp>/` contains `meta.json`, `result.md`, `result.zip` on host.
   - Re-upload same file → "instant from disk cache" message, no API traffic (watch `docker compose logs paddleocr-vl-api` — no new request).
   - Toggle one sidebar option → reprocesses into a second fingerprint subdir.
   - `docker compose down && up -d` → re-upload → still instant (persistence proves out).
   - Truncate `result.zip` → re-upload → reprocesses (corrupt-entry path).
   - Download buttons + batch ZIP still produce correct archives.
3. `docker compose config` validates compose syntax after edits.

## Out of scope

- Cache eviction/LRU, quota limits, metrics, multi-user isolation, storing raw API responses/visualizations on disk.
