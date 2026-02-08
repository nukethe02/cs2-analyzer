#!/usr/bin/env python3
"""
Full endpoint smoke test — hits every API endpoint using the golden demo.
Produces a summary table for auditing.
"""

import json
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

BASE = "http://127.0.0.1:7860"
DEMO_PATH = Path("tests/fixtures/golden_master.dem")
TIMEOUT = 30

# We'll populate these after uploading the demo
JOB_ID = None
STEAM_ID = None
STEAM_ID_B = None
DEMO_ID = None

results = []


def req(method, path, body=None, content_type=None, files=None, timeout=TIMEOUT):
    """Make an HTTP request, return (status, body_text, elapsed_ms)."""
    url = BASE + path
    t0 = time.perf_counter()
    try:
        if files:
            # Multipart form upload
            boundary = "----SmokeTestBoundary"
            lines = []
            for fname, fpath in files.items():
                data = Path(fpath).read_bytes()
                lines.append(f"--{boundary}".encode())
                lines.append(
                    f'Content-Disposition: form-data; name="{fname}"; filename="{Path(fpath).name}"'.encode()
                )
                lines.append(b"Content-Type: application/octet-stream")
                lines.append(b"")
                lines.append(data)
            lines.append(f"--{boundary}--".encode())
            body_bytes = b"\r\n".join(lines)
            r = urllib.request.Request(url, data=body_bytes, method=method)
            r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        elif body is not None:
            body_bytes = json.dumps(body).encode() if isinstance(body, (dict, list)) else body.encode()
            r = urllib.request.Request(url, data=body_bytes, method=method)
            r.add_header("Content-Type", content_type or "application/json")
        else:
            r = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(r, timeout=timeout) as resp:
            elapsed = (time.perf_counter() - t0) * 1000
            text = resp.read().decode("utf-8", errors="replace")
            return resp.status, text, elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        text = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return e.code, text, elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return 0, f"CONNECTION ERROR: {e}", elapsed


def has_data(status, body):
    """Check if response has meaningful data."""
    if status >= 400:
        return False
    try:
        j = json.loads(body)
        if isinstance(j, dict):
            if j.get("detail") and status >= 400:
                return False
            if j.get("error"):
                return False
            # Check if it's not just an empty structure
            if all(v is None or v == [] or v == {} or v == "" for v in j.values()):
                return False
        if isinstance(j, list) and len(j) == 0:
            return False
        return True
    except (json.JSONDecodeError, ValueError):
        # Non-JSON response (HTML, CSV, etc.)
        return len(body.strip()) > 10


def snippet(body, max_len=200):
    """First max_len chars of body."""
    s = body.strip().replace("\n", " ").replace("\r", "")
    return s[:max_len] + ("..." if len(s) > max_len else "")


def test(method, path, body=None, content_type=None, files=None, label=None, timeout=TIMEOUT):
    """Run one endpoint test and record the result."""
    display = label or f"{method} {path}"
    status, resp_body, elapsed = req(method, path, body, content_type, files, timeout)
    data = has_data(status, resp_body)
    flag = ""
    if status == 404:
        flag = "!! 404"
    elif status >= 500:
        flag = "!! 500+"
    elif not data:
        flag = "! NO DATA"
    elif elapsed > 5000:
        flag = "! SLOW"

    results.append({
        "endpoint": display,
        "status": status,
        "has_data": data,
        "time_ms": round(elapsed),
        "snippet": snippet(resp_body),
        "flag": flag,
    })
    marker = "PASS" if status < 400 and data else "FLAG" if flag else "WARN"
    print(f"  [{marker}] {status:>3}  {elapsed:>7.0f}ms  {display}")
    return status, resp_body, elapsed


def main():
    global JOB_ID, STEAM_ID, STEAM_ID_B, DEMO_ID

    print("=" * 80)
    print("OPENSIGHT FULL ENDPOINT SMOKE TEST")
    print("=" * 80)

    # ── Health & readiness ──────────────────────────────────────────
    print("\n-- Infrastructure Endpoints --")
    test("GET", "/health")
    test("GET", "/readiness")
    test("GET", "/about")
    test("GET", "/", label="GET / (HTML UI)")

    # ── Upload golden demo ──────────────────────────────────────────
    print("\n-- Demo Upload (POST /analyze) --")
    status, body, _ = test("POST", "/analyze", files={"file": str(DEMO_PATH)}, timeout=120)
    if status in (200, 202):
        j = json.loads(body)
        JOB_ID = j.get("job_id")
        print(f"     -> job_id = {JOB_ID}")
    else:
        print(f"     !! Upload failed with {status}, cannot test downstream endpoints")
        JOB_ID = "fake-job-id-for-testing"

    # ── Wait for job completion ─────────────────────────────────────
    if JOB_ID and JOB_ID != "fake-job-id-for-testing":
        print(f"\n-- Waiting for job {JOB_ID} to complete --")
        for i in range(60):
            status, body, _ = req("GET", f"/analyze/{JOB_ID}")
            if status == 200:
                j = json.loads(body)
                st = j.get("status", "")
                if st == "complete":
                    print(f"     -> Job complete after {i + 1} polls")
                    # Extract a steam_id from result
                    result_data = j.get("result", {})
                    players = result_data.get("players", {})
                    if players:
                        sids = list(players.keys())
                        STEAM_ID = sids[0]
                        STEAM_ID_B = sids[1] if len(sids) > 1 else sids[0]
                        DEMO_ID = JOB_ID  # demo_id is typically job_id
                        print(f"     -> steam_id_a = {STEAM_ID}")
                        print(f"     -> steam_id_b = {STEAM_ID_B}")
                    break
                elif st == "failed":
                    print(f"     !! Job failed: {j.get('error', 'unknown')}")
                    break
            time.sleep(1)
        else:
            print("     !! Job did not complete within 60s")

    if not STEAM_ID:
        STEAM_ID = "76561198199520810"
        STEAM_ID_B = "76561198240042038"
        DEMO_ID = JOB_ID or "test"
        print(f"     -> Using fallback steam_id = {STEAM_ID}")

    # ── Job status endpoints ────────────────────────────────────────
    print("\n-- Job Endpoints --")
    test("GET", f"/analyze/{JOB_ID}", label=f"GET /analyze/{{job_id}}")
    test("GET", f"/analyze/{JOB_ID}/download", label=f"GET /analyze/{{job_id}}/download")
    test("GET", "/jobs")

    # ── Export endpoints ────────────────────────────────────────────
    print("\n-- Export Endpoints --")
    test("GET", f"/api/export/{JOB_ID}/json", label="GET /api/export/{job_id}/json")
    test("GET", f"/api/export/{JOB_ID}/players-csv", label="GET /api/export/{job_id}/players-csv")
    test("GET", f"/api/export/{JOB_ID}/rounds-csv", label="GET /api/export/{job_id}/rounds-csv")

    # ── Heatmap endpoints ───────────────────────────────────────────
    print("\n-- Heatmap Endpoints --")
    test("GET", f"/api/heatmap/{JOB_ID}/kills", label="GET /api/heatmap/{job_id}/kills")
    test("GET", f"/api/heatmap/{JOB_ID}/grenades", label="GET /api/heatmap/{job_id}/grenades")

    # ── Your-match endpoints ────────────────────────────────────────
    print("\n-- Your-Match Endpoints --")
    test("GET", f"/api/your-match/{DEMO_ID}/{STEAM_ID}",
         label="GET /api/your-match/{demo_id}/{steam_id}")
    test("POST", "/api/your-match/store",
         body={"demo_id": DEMO_ID, "steam_id": STEAM_ID, "job_id": JOB_ID},
         label="POST /api/your-match/store")
    test("GET", f"/api/your-match/baselines/{STEAM_ID}",
         label="GET /api/your-match/baselines/{steam_id}")
    test("GET", f"/api/your-match/history/{STEAM_ID}",
         label="GET /api/your-match/history/{steam_id}")
    test("GET", f"/api/your-match/persona/{STEAM_ID}",
         label="GET /api/your-match/persona/{steam_id}")
    test("GET", f"/api/your-match/trends/{STEAM_ID}",
         label="GET /api/your-match/trends/{steam_id}")

    # ── Player metrics endpoint ─────────────────────────────────────
    print("\n-- Player Endpoints --")
    test("GET", f"/api/players/{STEAM_ID}/metrics",
         label="GET /api/players/{steam_id}/metrics")

    # ── Positioning endpoints ───────────────────────────────────────
    print("\n-- Positioning Endpoints --")
    test("GET", f"/api/positioning/{JOB_ID}/{STEAM_ID}",
         label="GET /api/positioning/{job_id}/{steam_id}")
    test("GET", f"/api/positioning/{JOB_ID}/compare/{STEAM_ID}/{STEAM_ID_B}",
         label="GET /api/positioning/{job_id}/compare/{a}/{b}")
    test("GET", f"/api/positioning/{JOB_ID}/all",
         label="GET /api/positioning/{job_id}/all")

    # ── Trade chains ────────────────────────────────────────────────
    print("\n-- Trade Chain Endpoints --")
    test("GET", f"/api/trade-chains/{JOB_ID}",
         label="GET /api/trade-chains/{job_id}")

    # ── AI endpoints ────────────────────────────────────────────────
    print("\n-- AI Endpoints (may fail without ANTHROPIC_API_KEY) --")
    test("POST", f"/api/tactical-analysis/{JOB_ID}",
         body={}, label="POST /api/tactical-analysis/{job_id}")
    test("POST", f"/api/strat-steal/{JOB_ID}",
         body={}, label="POST /api/strat-steal/{job_id}")
    test("POST", f"/api/self-review/{JOB_ID}",
         body={}, label="POST /api/self-review/{job_id}")

    # ── Maps endpoints ──────────────────────────────────────────────
    print("\n-- Maps Endpoints --")
    test("GET", "/maps")
    test("GET", "/maps/de_ancient", label="GET /maps/{map_name}")
    test("POST", "/radar/transform",
         body={"map_name": "de_ancient", "x": 100.0, "y": 200.0, "z": 0.0},
         label="POST /radar/transform")

    # ── HLTV endpoints ──────────────────────────────────────────────
    print("\n-- HLTV Endpoints --")
    test("GET", "/hltv/rankings")
    test("GET", "/hltv/map/de_ancient", label="GET /hltv/map/{map_name}")
    test("GET", "/hltv/player/search?nickname=s1mple",
         label="GET /hltv/player/search")
    test("POST", "/hltv/enrich",
         body={"job_id": JOB_ID},
         label="POST /hltv/enrich")

    # ── Replay endpoint ─────────────────────────────────────────────
    print("\n-- Replay Endpoint --")
    test("POST", "/replay/generate",
         files={"file": str(DEMO_PATH)},
         label="POST /replay/generate", timeout=120)

    # ── Tactical rotations ──────────────────────────────────────────
    print("\n-- Tactical Rotations --")
    test("POST", "/tactical/rotations",
         files={"file": str(DEMO_PATH)},
         label="POST /tactical/rotations", timeout=120)

    # ── Decode share code ───────────────────────────────────────────
    print("\n-- Decode Endpoint --")
    test("POST", "/decode",
         body={"share_code": "CSGO-xxxxx-xxxxx-xxxxx-xxxxx-xxxxx"},
         label="POST /decode")

    # ── Cache endpoints ─────────────────────────────────────────────
    print("\n-- Cache Endpoints --")
    test("GET", "/cache/stats")
    test("POST", "/cache/clear", body={})

    # ── Feedback endpoints ──────────────────────────────────────────
    print("\n-- Feedback Endpoints --")
    test("POST", "/feedback",
         body={"metric": "hltv_rating", "expected": 1.2, "actual": 1.1, "comment": "test"},
         label="POST /feedback")
    test("POST", "/feedback/coaching",
         body={"job_id": JOB_ID, "steam_id": STEAM_ID, "rating": 4, "comment": "smoke test"},
         label="POST /feedback/coaching")
    test("GET", "/feedback/stats")

    # ── Parallel status ─────────────────────────────────────────────
    print("\n-- Parallel Endpoints --")
    test("GET", "/parallel/status")

    # ── Scouting endpoints ──────────────────────────────────────────
    print("\n-- Scouting Endpoints --")
    status, body, _ = test("POST", "/api/scouting/session",
                           body={"team_name": "TestTeam", "map_pool": ["de_ancient"]},
                           label="POST /api/scouting/session")
    scouting_session_id = None
    if status in (200, 201):
        try:
            scouting_session_id = json.loads(body).get("session_id")
        except Exception:
            pass

    if scouting_session_id:
        test("GET", f"/api/scouting/session/{scouting_session_id}",
             label="GET /api/scouting/session/{session_id}")
        test("POST", f"/api/scouting/session/{scouting_session_id}/add-demo",
             files={"file": str(DEMO_PATH)},
             label="POST /api/scouting/session/{id}/add-demo", timeout=120)
        test("POST", f"/api/scouting/session/{scouting_session_id}/report",
             body={},
             label="POST /api/scouting/session/{id}/report")
        test("DELETE", f"/api/scouting/session/{scouting_session_id}",
             label="DELETE /api/scouting/session/{id}")
    else:
        # Test with a fake session_id to verify the endpoint exists
        test("GET", "/api/scouting/session/fake-session",
             label="GET /api/scouting/session/{session_id}")
        test("DELETE", "/api/scouting/session/fake-session",
             label="DELETE /api/scouting/session/{id}")

    # ── Auth endpoints ──────────────────────────────────────────────
    print("\n-- Auth Endpoints --")
    test("POST", "/auth/register",
         body={"username": "smoketest", "password": "testpass123", "email": "test@test.com"},
         label="POST /auth/register")
    test("POST", "/auth/login",
         body={"username": "smoketest", "password": "testpass123"},
         label="POST /auth/login")
    test("GET", "/auth/me", label="GET /auth/me")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = f"{'ENDPOINT':<55} {'STATUS':>6} {'DATA':>6} {'MS':>7}  {'NOTES'}"
    print(header)
    print("-" * 120)

    flagged = []
    for r in results:
        data_str = "YES" if r["has_data"] else "NO"
        line = f"{r['endpoint']:<55} {r['status']:>6} {data_str:>6} {r['time_ms']:>7}  {r['flag']}"
        print(line)
        if r["flag"]:
            flagged.append(r)

    print("-" * 120)
    total = len(results)
    passed = sum(1 for r in results if r["status"] < 400 and r["has_data"])
    print(f"\nTOTAL: {total} endpoints tested")
    print(f"PASSED: {passed}/{total} ({100*passed/total:.0f}%)")
    print(f"FLAGGED: {len(flagged)}")

    if flagged:
        print("\n" + "=" * 120)
        print("FLAGGED ENDPOINTS (detail)")
        print("=" * 120)
        for r in flagged:
            print(f"\n  {r['flag']} -> {r['endpoint']}")
            print(f"    Status: {r['status']}, Time: {r['time_ms']}ms")
            print(f"    Body: {r['snippet']}")

    # ── Response snippets ───────────────────────────────────────────
    print("\n" + "=" * 120)
    print("RESPONSE SNIPPETS (first 200 chars per endpoint)")
    print("=" * 120)
    for r in results:
        print(f"\n  [{r['status']}] {r['endpoint']}")
        print(f"    {r['snippet']}")

    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
