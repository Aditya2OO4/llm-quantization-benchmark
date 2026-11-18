import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import step6_report
import step7_load_test as lt
import step11_serving_compare as sc


def test_payload_formats():
    fast = lt.build_payload("fastapi", "hi", 32)
    assert fast == {"prompt": "hi", "max_tokens": 32, "temperature": 0.0}  # unchanged default behaviour
    srv = lt.build_payload("llamaserver", "hi", 32)
    assert srv["n_predict"] == 32 and srv["cache_prompt"] is False and "max_tokens" not in srv


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/generate":
            out = {"tokens_generated": body["max_tokens"]}
        elif self.path == "/completion":
            out = {"tokens_predicted": body["n_predict"]}
        else:
            self.send_error(404)
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.mark.parametrize("fmt", ["fastapi", "llamaserver"])
def test_load_test_counts_tokens_for_each_format(server, fmt):
    payload = lt.build_payload(fmt, "hi", 7)
    row = lt.run_level(server, 2, 4, payload, 10, fmt)
    assert row["requests"] == 4 and row["tokens_per_s"] > 0 and row["p95_s"] >= row["p50_s"]
    one = lt.one_request(server, payload, 10, fmt)
    assert one["tokens"] == 7


def test_one_request_default_is_still_fastapi(server):
    assert lt.one_request(server, {"prompt": "x", "max_tokens": 3}, 10)["tokens"] == 3


def test_llamaserver_command_gives_each_slot_its_own_context(monkeypatch, tmp_path):
    (tmp_path / "llama-server.exe").write_text("")
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    cmd = sc.llamaserver_cmd(Path("m.gguf"), 8090, 4, 8)
    assert cmd[cmd.index("-np") + 1] == "8" and "-cb" in cmd
    assert cmd[cmd.index("-c") + 1] == str(8 * sc.CTX_PER_SLOT)


def test_wait_until_ready_reports_early_exit():
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
    proc.wait()
    with pytest.raises(RuntimeError, match="exited early"):
        sc.wait_until_ready("http://127.0.0.1:1", "/health", proc, timeout=5)


def _lvl(c, tps):
    return {"concurrency": c, "req_per_s": 1.0, "tokens_per_s": tps, "p50_s": 2.0, "p95_s": 3.0}


def test_serving_table_pairs_levels():
    fast = {"threads": 4, "max_tokens": 32, "requests_per_level": 24, "levels": [_lvl(1, 27.0), _lvl(4, 30.0)]}
    llama = {"slots": 8, "levels": [_lvl(1, 25.0), _lvl(4, 40.0), _lvl(8, 41.0)]}
    md = "\n".join(step6_report.serving_table(fast, llama))
    assert "| 1 | 1.00 | 27.0 |" in md and "| 4 | 1.00 | 30.0 |" in md and "| 8 |" not in md
    assert step6_report.serving_table(fast, None) == []
    ctrl = {"slots": 1, "levels": [_lvl(1, 20.0), _lvl(4, 21.0)]}
    with_ctrl = "\n".join(step6_report.serving_table(fast, llama, ctrl))
    assert "1 slot (control)" in with_ctrl and "| 4 | 1.00 | 30.0 | 2.00 | 3.00 | 1.00 | 40.0 | 2.00 | 3.00 | 1.00 | 21.0 |" in with_ctrl
