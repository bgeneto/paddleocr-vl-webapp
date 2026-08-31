#!/usr/bin/env python3
"""Reverse-proxy paddlex --serve and stop it after idle so GPU memory is released.

PaddleX has no llama-server --sleep-idle-seconds equivalent. PP-DocLayoutV3 (and
quality-first extras) stay resident, and Paddle's CUDA allocator does not return
that pool to the driver — nvidia-smi keeps showing ~4 GiB. The only reliable
release is to exit the paddlex process.

This proxy:
  - Listens on :8080 (same URL Streamlit already uses)
  - Runs paddlex on 127.0.0.1:8081
  - GET/HEAD /health never starts paddlex and never resets the idle timer
    (Docker healthchecks would otherwise pin the models in VRAM forever)
  - Stops paddlex after PADDLEX_SLEEP_IDLE_SECONDS with no in-flight work
  - Starts it again on the next real request

Set PADDLEX_SLEEP_IDLE_SECONDS=-1 to keep paddlex running (old behavior).
"""

import argparse
import http.client
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    }
)
IDLE_SAFE_PATHS = frozenset({"/health", "/healthz", "/ready", "/readyz"})
CHUNK = 64 * 1024


def log(msg):
    print("[paddlex-idle] " + msg, flush=True)


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


class Backend:
    def __init__(self, args):
        self.args = args
        self.cond = threading.Condition()
        self.proc = None
        self.in_flight = 0
        self.starting = False
        self.last_idle = time.monotonic()
        self.idle_seconds = env_int("PADDLEX_SLEEP_IDLE_SECONDS", 60)
        self.ready_timeout = env_int("PADDLEX_BACKEND_READY_TIMEOUT", 180)

    def _is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def _paddlex_cmd(self):
        cmd = [
            os.environ.get("PADDLEX_BIN", "paddlex"),
            "--serve",
            "--pipeline",
            self.args.pipeline,
            "--host",
            self.args.backend_host,
            "--port",
            str(self.args.backend_port),
        ]
        device = os.environ.get("PADDLEX_DEVICE", "").strip()
        if device:
            cmd.extend(["--device", device])
        return cmd

    def _wait_healthy(self, proc):
        deadline = time.monotonic() + self.ready_timeout
        last_err = "timeout"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "paddlex exited with code %s before becoming healthy"
                    % (proc.returncode,)
                )
            conn = None
            try:
                conn = http.client.HTTPConnection(
                    self.args.backend_host, self.args.backend_port, timeout=2
                )
                conn.request("GET", "/health")
                resp = conn.getresponse()
                resp.read()
                if resp.status == 200:
                    return
                last_err = "HTTP %s" % resp.status
            except OSError as exc:
                last_err = str(exc)
            finally:
                if conn is not None:
                    conn.close()
            time.sleep(1)
        raise RuntimeError(
            "paddlex not healthy on %s:%s within %ss (%s)"
            % (
                self.args.backend_host,
                self.args.backend_port,
                self.ready_timeout,
                last_err,
            )
        )

    def _stop_locked(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        if proc.poll() is not None:
            log("paddlex already exited with code %s" % proc.returncode)
            return
        log("stopping paddlex (pid %s) to release GPU memory" % proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError as exc:
            log("SIGTERM failed: %s" % exc)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log("paddlex did not exit in 15s; sending SIGKILL")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
        log("paddlex stopped")

    def _spawn_locked(self):
        cmd = self._paddlex_cmd()
        log("starting: %s" % " ".join(cmd))
        self.proc = subprocess.Popen(cmd, start_new_session=True)
        return self.proc

    def _start_backend(self, bump_in_flight):
        """Start paddlex if needed. Does not hold the lock while waiting on /health.

        When bump_in_flight is True, in_flight is incremented before the lock is
        dropped so the idle watcher cannot kill the process between start and use.
        """
        with self.cond:
            while self.starting:
                self.cond.wait()
            if self._is_running():
                if bump_in_flight:
                    self.in_flight += 1
                return
            self.starting = True
            try:
                proc = self._spawn_locked()
            except Exception:
                self.starting = False
                self.cond.notify_all()
                raise
        try:
            self._wait_healthy(proc)
        except Exception:
            with self.cond:
                self._stop_locked()
                self.starting = False
                self.cond.notify_all()
            raise
        with self.cond:
            self.starting = False
            self.last_idle = time.monotonic()
            if bump_in_flight:
                self.in_flight += 1
            self.cond.notify_all()
        log(
            "paddlex ready on %s:%s (pid %s)"
            % (self.args.backend_host, self.args.backend_port, proc.pid)
        )

    def ensure_started(self):
        self._start_backend(bump_in_flight=False)

    def begin_request(self):
        self._start_backend(bump_in_flight=True)

    def end_request(self):
        with self.cond:
            self.in_flight = max(0, self.in_flight - 1)
            if self.in_flight == 0:
                self.last_idle = time.monotonic()

    def idle_watch(self):
        if self.idle_seconds < 0:
            log("idle stop disabled (PADDLEX_SLEEP_IDLE_SECONDS=-1)")
            return
        log(
            "will stop paddlex after %ss with no in-flight requests"
            % self.idle_seconds
        )
        while True:
            time.sleep(1)
            with self.cond:
                if self.starting or not self._is_running() or self.in_flight > 0:
                    continue
                if time.monotonic() - self.last_idle >= self.idle_seconds:
                    self._stop_locked()

    def shutdown(self):
        with self.cond:
            self._stop_locked()
            self.starting = False
            self.cond.notify_all()


BACKEND = None


def is_idle_safe(handler):
    path = handler.path.split("?", 1)[0]
    return path in IDLE_SAFE_PATHS and handler.command in ("GET", "HEAD")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    close_connection = True

    def log_message(self, fmt, *args):
        if is_idle_safe(self):
            return
        sys.stderr.write("[paddlex-idle] " + (fmt % args) + "\n")

    def _send_health(self):
        body = b'{"status":"ok"}\n'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_plain(self, status, message):
        body = (message + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _proxy(self):
        backend = BACKEND
        started = False
        conn = None
        headers_sent = False
        try:
            backend.begin_request()
            started = True
            length_hdr = self.headers.get("Content-Length")
            remaining = int(length_hdr) if length_hdr else 0
            conn = http.client.HTTPConnection(
                backend.args.backend_host,
                backend.args.backend_port,
                timeout=3600,
            )
            conn.putrequest(
                self.command, self.path, skip_host=True, skip_accept_encoding=True
            )
            conn.putheader(
                "Host",
                "%s:%s" % (backend.args.backend_host, backend.args.backend_port),
            )
            conn.putheader("Connection", "close")
            for key, value in self.headers.items():
                lk = key.lower()
                if lk in HOP_BY_HOP or lk in ("host", "connection"):
                    continue
                conn.putheader(key, value)
            conn.endheaders()
            while remaining > 0:
                chunk = self.rfile.read(min(CHUNK, remaining))
                if not chunk:
                    break
                conn.send(chunk)
                remaining -= len(chunk)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                if key.lower() in HOP_BY_HOP:
                    continue
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            headers_sent = True
            if self.command != "HEAD":
                shutil.copyfileobj(resp, self.wfile, CHUNK)
        except Exception as exc:
            log("proxy error for %s %s: %s" % (self.command, self.path, exc))
            if not headers_sent:
                try:
                    self._send_plain(502, "paddlex backend error: %s" % exc)
                except Exception:
                    pass
        finally:
            if conn is not None:
                conn.close()
            if started:
                backend.end_request()

    def _dispatch(self):
        if is_idle_safe(self):
            self._send_health()
            return
        self._proxy()

    def do_GET(self):
        self._dispatch()

    def do_HEAD(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def do_PUT(self):
        self._dispatch()

    def do_DELETE(self):
        self._dispatch()

    def do_PATCH(self):
        self._dispatch()

    def do_OPTIONS(self):
        self._dispatch()


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pipeline", required=True, help="PaddleX pipeline YAML path")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8081)
    return parser.parse_args(argv)


def main(argv=None):
    global BACKEND
    args = parse_args(argv if argv is not None else sys.argv[1:])
    BACKEND = Backend(args)

    def on_signal(signum, _frame):
        log("caught signal %s" % signum)
        BACKEND.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    watcher = threading.Thread(
        target=BACKEND.idle_watch, name="paddlex-idle", daemon=True
    )
    watcher.start()

    # Load models at container start so the first OCR after `compose up` is warm.
    # If the stack then sits idle, the watcher releases VRAM.
    try:
        BACKEND.ensure_started()
    except Exception as exc:
        log("initial paddlex start failed (%s); will retry on first request" % exc)

    server = ThreadingHTTPServer((args.listen_host, args.listen_port), ProxyHandler)
    server.daemon_threads = True
    log(
        "proxy listening on %s:%s -> %s:%s"
        % (
            args.listen_host,
            args.listen_port,
            args.backend_host,
            args.backend_port,
        )
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        BACKEND.shutdown()


if __name__ == "__main__":
    main()
