"""Full round-trip integration test of the OpenSight server.

Starts the server, uploads a demo, polls for completion, hits endpoints, reports results.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PORT = 8765
BASE = f"http://localhost:{PORT}"
DEMO_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "golden_master.dem"
RESULTS = []
TOTAL_START = time.perf_counter()


def report(step: str, passed: bool, detail: str = "", elapsed: float = 0.0):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((step, passed, detail, elapsed))
    t = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
    print(f"  [{status}] {step}{t}" + (f" — {detail}" if detail else ""))


def fetch(url: str, method: str = "GET", timeout: int = 30) -> tuple[int, bytes]:
    req = Request(url, method=method)
    resp = urlopen(req, timeout=timeout)
    return resp.status, resp.read()


def upload_file(url: str, filepath: Path, timeout: int = 300) -> tuple[int, bytes]:
    """Multipart file upload using urllib."""
    import mimetypes
    import uuid

    boundary = uuid.uuid4().hex
    filename = filepath.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body = b""
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += f"Content-Type: {content_type}\r\n\r\n".encode()
    body += filepath.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    resp = urlopen(req, timeout=timeout)
    return resp.status, resp.read()


# ─── Step 0: Check demo file exists ───
if not DEMO_PATH.exists():
    print(f"FATAL: Golden demo not found at {DEMO_PATH}")
    sys.exit(1)
print(f"Demo: {DEMO_PATH} ({DEMO_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

# ─── Step 1: Start server ───
print("\n=== Step 1: Start server ===")
t0 = time.perf_counter()
server_proc = subprocess.Popen(
    [sys.executable, "-m", "opensight.server", "--port", str(PORT)],
    env={**__import__("os").environ, "PYTHONPATH": "src"},
    cwd=str(Path(__file__).parent.parent),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
report("Server process started", True, f"PID {server_proc.pid}", time.perf_counter() - t0)

# ─── Step 2: Wait for health ───
print("\n=== Step 2: Wait for /health ===")
t0 = time.perf_counter()
healthy = False
for attempt in range(30):
    try:
        status, body = fetch(f"{BASE}/health", timeout=3)
        if status == 200:
            healthy = True
            break
    except (URLError, OSError, ConnectionError):
        pass
    time.sleep(1)
elapsed = time.perf_counter() - t0
report("Health check", healthy, f"after {attempt + 1} attempts", elapsed)

if not healthy:
    print("Server failed to start. stderr:")
    server_proc.terminate()
    print(server_proc.stderr.read().decode(errors="replace")[-2000:])
    sys.exit(1)

# ─── Step 3: Upload golden demo ───
print("\n=== Step 3: Upload demo ===")
t0 = time.perf_counter()
try:
    status, body = upload_file(f"{BASE}/analyze", DEMO_PATH, timeout=60)
    upload_resp = json.loads(body)
    job_id = upload_resp.get("job_id", "")
    elapsed = time.perf_counter() - t0
    report("Upload demo", status in (200, 202) and bool(job_id), f"status={status}, job_id={job_id}", elapsed)
except Exception as e:
    elapsed = time.perf_counter() - t0
    report("Upload demo", False, str(e), elapsed)
    job_id = ""

if not job_id:
    print("Cannot continue without job_id.")
    server_proc.terminate()
    sys.exit(1)

# ─── Step 4: Poll for completion ───
print("\n=== Step 4: Poll for completion ===")
t0 = time.perf_counter()
final_status = "unknown"
poll_timeout = 300  # 5 minutes max
while (time.perf_counter() - t0) < poll_timeout:
    try:
        status, body = fetch(f"{BASE}/analyze/{job_id}", timeout=10)
        poll_resp = json.loads(body)
        final_status = poll_resp.get("status", "unknown")
        if final_status in ("completed", "complete", "failed"):
            break
    except Exception:
        pass
    time.sleep(3)

elapsed = time.perf_counter() - t0
completed = final_status in ("completed", "complete")
if not completed:
    detail = f"status={final_status}"
    if final_status == "failed":
        detail += f", error={poll_resp.get('error', 'N/A')}"
    report("Analysis complete", False, detail, elapsed)
else:
    report("Analysis complete", True, f"status={final_status}", elapsed)

# ─── Step 5: Hit download endpoint ───
print("\n=== Step 5: Hit endpoints ===")

endpoints = [
    ("GET /analyze/{job_id}/download", f"/analyze/{job_id}/download"),
    ("GET /api/export/{job_id}/json", f"/api/export/{job_id}/json"),
    ("GET /api/export/{job_id}/players-csv", f"/api/export/{job_id}/players-csv"),
    ("GET /api/heatmap/{job_id}/kills", f"/api/heatmap/{job_id}/kills"),
]

for name, path in endpoints:
    t0 = time.perf_counter()
    try:
        status, body = fetch(f"{BASE}{path}", timeout=30)
        elapsed = time.perf_counter() - t0
        ok = status == 200 and len(body) > 0
        detail = f"status={status}, size={len(body)} bytes"
        # Quick sanity on JSON endpoints
        if ok and path.endswith("/json"):
            data = json.loads(body)
            n_players = len(data.get("players", {}))
            detail += f", {n_players} players"
        elif ok and path.endswith("/download"):
            data = json.loads(body)
            # Try both nested and flat structures
            players = data.get("result", {}).get("players", {}) or data.get("players", {})
            n_players = len(players)
            detail += f", {n_players} players in result"
        elif ok and path.endswith("/kills"):
            data = json.loads(body)
            n_kills = len(data.get("kills", []))
            detail += f", {n_kills} kills"
        elif ok and path.endswith("/players-csv"):
            lines = body.decode(errors="replace").strip().split("\n")
            detail += f", {len(lines)} lines (header + {len(lines)-1} players)"
        report(name, ok, detail, elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        report(name, False, str(e), elapsed)

# ─── Step 6: Bonus endpoints ───
print("\n=== Step 6: Bonus endpoints ===")
bonus = [
    ("GET /health", "/health"),
    ("GET /readiness", "/readiness"),
    ("GET /jobs", "/jobs"),
    ("GET /maps", "/maps"),
    ("GET /about", "/about"),
]
for name, path in bonus:
    t0 = time.perf_counter()
    try:
        status, body = fetch(f"{BASE}{path}", timeout=10)
        elapsed = time.perf_counter() - t0
        report(name, status == 200 and len(body) > 0, f"status={status}, {len(body)} bytes", elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        report(name, False, str(e), elapsed)

# ─── Step 7: Kill server ───
print("\n=== Step 7: Shutdown ===")
server_proc.terminate()
try:
    server_proc.wait(timeout=5)
    report("Server shutdown", True, f"exit code {server_proc.returncode}")
except subprocess.TimeoutExpired:
    server_proc.kill()
    report("Server shutdown", True, "force killed")

# ─── Report ───
total_elapsed = time.perf_counter() - TOTAL_START
print("\n" + "=" * 70)
print("INTEGRATION TEST REPORT")
print("=" * 70)
passed = sum(1 for _, ok, _, _ in RESULTS if ok)
failed = sum(1 for _, ok, _, _ in RESULTS if not ok)
for step, ok, detail, elapsed in RESULTS:
    icon = "+" if ok else "X"
    t = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
    print(f"  [{icon}] {step}{t}")
    if detail and not ok:
        print(f"       {detail}")
print(f"\nTotal: {passed} passed, {failed} failed, {total_elapsed:.1f}s wall time")
sys.exit(0 if failed == 0 else 1)
