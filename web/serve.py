"""Zero-dependency local HTTP server for HEARSAY Forensic Audio Workstation.

Usage:
  uv run python web/serve.py [--port 8000]
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent


class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers for local testing
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve HEARSAY Forensic UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    port = args.port
    # Retry on next ports if 8000 is occupied
    for attempt in range(10):
        try:
            with socketserver.TCPServer(("", port), CustomHandler) as httpd:
                url = f"http://localhost:{port}/index.html"
                print("\n=======================================================")
                print("  HEARSAY Forensic Workstation running at:")
                print(f"  --> {url}")
                print("=======================================================\n")
                if not args.no_browser:
                    webbrowser.open(url)
                httpd.serve_forever()
        except OSError:
            if attempt < 9:
                port += 1
            else:
                raise


if __name__ == "__main__":
    main()
