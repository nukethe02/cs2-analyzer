"""
Smoke test - Comprehensive import verification for all OpenSight modules.
Run with: PYTHONPATH=src python scripts/smoke_test.py
"""

import sys
import importlib

MODULES = [
    "opensight.api",
    "opensight.core.parser",
    "opensight.core.config",
    "opensight.core.constants",
    "opensight.core.utils",
    "opensight.core.map_zones",
    "opensight.analysis.analytics",
    "opensight.analysis.models",
    "opensight.analysis.compute_combat",
    "opensight.analysis.compute_aim",
    "opensight.analysis.compute_economy",
    "opensight.analysis.compute_utility",
    "opensight.analysis.highlights",
    "opensight.analysis.hltv_rating",
    "opensight.analysis.persona",
    "opensight.analysis.positioning",
    "opensight.analysis.rotation",
    "opensight.pipeline.orchestrator",
    "opensight.pipeline.store_events",
    "opensight.infra.database",
    "opensight.infra.cache",
    "opensight.infra.job_store",
    "opensight.auth.passwords",
    "opensight.auth.jwt",
    "opensight.auth.tiers",
    "opensight.visualization.heatmaps",
    "opensight.visualization.exports",
    "opensight.visualization.radar",
    "opensight.visualization.replay",
    "opensight.domains.combat",
    "opensight.domains.economy",
    "opensight.domains.synergy",
    "opensight.scouting.engine",
    "opensight.ai.llm_client",
    "opensight.integrations.feedback",
]

passed = 0
failed = 0

print("=" * 60)
print("OpenSight Smoke Test - Module Import Verification")
print("=" * 60)

for mod in MODULES:
    try:
        importlib.import_module(mod)
        print(f"  ✓ {mod}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {mod} — {e}")
        failed += 1

# Check app specifically
print("\n" + "=" * 60)
print("API App Verification")
print("=" * 60)

try:
    from opensight.api import app, job_store, sharecode_cache

    routes = [r for r in app.routes if hasattr(r, "methods")]
    print(f"  ✓ App loaded: {len(routes)} endpoints registered")
    print(f"  ✓ job_store: {type(job_store).__name__}")
    print(f"  ✓ sharecode_cache: {type(sharecode_cache).__name__}")
    passed += 1
except Exception as e:
    print(f"  ✗ App failed: {e}")
    failed += 1

# Summary
print("\n" + "=" * 60)
print(f"✓ {passed} passed, ✗ {failed} failed")
print("=" * 60)

sys.exit(1 if failed else 0)
