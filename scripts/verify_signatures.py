"""
Verify cross-module call signatures and type annotation quality.

Detects:
  1. Signature mismatches — wrong kwargs, missing required positional args
  2. Missing return annotations on public functions
  3. Any propagation — explicit Any in signatures that should be concrete
  4. Unsafe attribute access — getattr without default on optional values

Usage:
    PYTHONPATH=src python scripts/verify_signatures.py

Exit codes:
    0 - No issues found
    1 - Issues detected
"""

import ast
import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_ROOT = Path(__file__).parent.parent
SCAN_DIRS = [
    PROJECT_ROOT / "src" / "opensight",
    PROJECT_ROOT / "scripts",
]

# Modules where we skip annotation checks (too large, pre-existing debt)
ANNOTATION_SKIP = {"analytics.py"}


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class Issue:
    file_path: str
    lineno: int
    category: str  # "signature" | "annotation" | "any-propagation" | "unsafe-access"
    severity: str  # "error" | "warning"
    message: str
    source_line: str = ""


@dataclass
class ImportInfo:
    local_name: str
    module_path: str
    attr_name: str
    lineno: int


@dataclass
class CallSite:
    file_path: str
    lineno: int
    func_name: str
    num_positional: int
    keyword_names: list[str]
    has_starargs: bool
    has_starkwargs: bool
    source_line: str


# =============================================================================
# Phase 1: File collection
# =============================================================================


def collect_files() -> list[Path]:
    files = []
    for d in SCAN_DIRS:
        if d.exists():
            files.extend(sorted(d.rglob("*.py")))
    return files


# =============================================================================
# Phase 2: AST helpers
# =============================================================================


def _annotate_parents(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def _is_inside_function(node: ast.AST) -> bool:
    current = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
    return False


def parse_imports(tree: ast.Module) -> list[ImportInfo]:
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("opensight"):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        local_name=alias.asname or alias.name,
                        module_path=node.module,
                        attr_name=alias.name,
                        lineno=node.lineno,
                    )
                )
    return imports


def collect_calls(
    tree: ast.Module, import_names: set[str], file_path: str, source_lines: list[str]
) -> list[CallSite]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in import_names:
            continue

        has_starargs = any(isinstance(a, ast.Starred) for a in node.args)
        has_starkwargs = any(kw.arg is None for kw in node.keywords)

        calls.append(
            CallSite(
                file_path=file_path,
                lineno=node.lineno,
                func_name=name,
                num_positional=len([a for a in node.args if not isinstance(a, ast.Starred)]),
                keyword_names=[kw.arg for kw in node.keywords if kw.arg is not None],
                has_starargs=has_starargs,
                has_starkwargs=has_starkwargs,
                source_line=source_lines[node.lineno - 1].strip()
                if node.lineno <= len(source_lines)
                else "",
            )
        )
    return calls


# =============================================================================
# Phase 3: Signature resolution + checking
# =============================================================================

_sig_cache: dict[str, tuple[inspect.Signature | None, bool]] = {}


def resolve_signature(imp: ImportInfo) -> tuple[inspect.Signature | None, bool]:
    fqn = f"{imp.module_path}.{imp.attr_name}"
    if fqn in _sig_cache:
        return _sig_cache[fqn]

    try:
        mod = importlib.import_module(imp.module_path)
        attr = getattr(mod, imp.attr_name, None)
        if attr is None:
            _sig_cache[fqn] = (None, False)
            return None, False

        is_class = inspect.isclass(attr)
        target = attr.__init__ if is_class else attr  # type: ignore[misc]

        if is_class and (target is object.__init__):
            _sig_cache[fqn] = (None, True)
            return None, True

        sig = inspect.signature(target)
        _sig_cache[fqn] = (sig, is_class)
        return sig, is_class
    except Exception:
        _sig_cache[fqn] = (None, False)
        return None, False


def check_signature(
    call: CallSite, sig: inspect.Signature, is_class: bool, fqn: str
) -> list[Issue]:
    issues: list[Issue] = []
    params = list(sig.parameters.values())

    if is_class and params and params[0].name == "self":
        params = params[1:]

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    # Check 1: unknown keyword arguments
    if not has_var_kw and not call.has_starkwargs:
        valid_names = {p.name for p in params}
        for kw in call.keyword_names:
            if kw not in valid_names:
                issues.append(
                    Issue(
                        file_path=call.file_path,
                        lineno=call.lineno,
                        category="signature",
                        severity="error",
                        message=(f"Unknown kwarg '{kw}' on {fqn}. Valid: {sorted(valid_names)}"),
                        source_line=call.source_line,
                    )
                )

    # Check 2: missing required positional arguments
    if not has_var_pos and not call.has_starargs:
        required = []
        for p in params:
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if p.default is inspect.Parameter.empty and p.name not in call.keyword_names:
                    required.append(p.name)
            elif p.kind == inspect.Parameter.KEYWORD_ONLY:
                break

        if call.num_positional < len(required):
            missing = required[call.num_positional :]
            issues.append(
                Issue(
                    file_path=call.file_path,
                    lineno=call.lineno,
                    category="signature",
                    severity="error",
                    message=(
                        f"Missing required arg(s) {missing} on {fqn}. "
                        f"Got {call.num_positional} positional, need {len(required)}"
                    ),
                    source_line=call.source_line,
                )
            )

    return issues


# =============================================================================
# Phase 4: Annotation quality checks
# =============================================================================


def _annotation_is_any(node: ast.expr | None) -> bool:
    """Check if an annotation AST node is literally 'Any'."""
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == "Any":
        return True
    if isinstance(node, ast.Constant) and node.value == "Any":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Any":
        return True
    return False


def check_annotations(tree: ast.Module, file_path: str, source_lines: list[str]) -> list[Issue]:
    """Check for missing return annotations and Any propagation."""
    issues: list[Issue] = []
    filename = Path(file_path).name

    if filename in ANNOTATION_SKIP:
        return issues

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Skip private/dunder methods and test helpers
        if node.name.startswith("_"):
            continue

        # Skip nested functions (closures, callbacks)
        if _is_inside_function(node):
            continue

        line = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""

        # Check: missing return annotation
        if node.returns is None:
            issues.append(
                Issue(
                    file_path=file_path,
                    lineno=node.lineno,
                    category="annotation",
                    severity="warning",
                    message=f"Public function '{node.name}' has no return type annotation",
                    source_line=line,
                )
            )

        # Check: return type is explicitly Any
        if _annotation_is_any(node.returns):
            issues.append(
                Issue(
                    file_path=file_path,
                    lineno=node.lineno,
                    category="any-propagation",
                    severity="warning",
                    message=f"Function '{node.name}' returns 'Any' — consider a concrete type",
                    source_line=line,
                )
            )

        # Check: parameters explicitly typed as Any
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            if arg.arg == "self":
                continue
            if _annotation_is_any(arg.annotation):
                issues.append(
                    Issue(
                        file_path=file_path,
                        lineno=arg.lineno if hasattr(arg, "lineno") else node.lineno,
                        category="any-propagation",
                        severity="warning",
                        message=(
                            f"Parameter '{arg.arg}' in '{node.name}' "
                            f"is typed as 'Any' — consider a concrete type"
                        ),
                        source_line=line,
                    )
                )

    return issues


# =============================================================================
# Phase 5: Unsafe access patterns
# =============================================================================


def check_unsafe_access(tree: ast.Module, file_path: str, source_lines: list[str]) -> list[Issue]:
    """Detect getattr() calls without a default value."""
    issues: list[Issue] = []

    for node in ast.walk(tree):
        # getattr(obj, "name") without a default — will raise AttributeError
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 2
            and not node.keywords
        ):
            line = source_lines[node.lineno - 1].strip() if node.lineno <= len(source_lines) else ""
            # Skip if inside a try/except block
            if not _is_inside_try(node):
                issues.append(
                    Issue(
                        file_path=file_path,
                        lineno=node.lineno,
                        category="unsafe-access",
                        severity="warning",
                        message="getattr() without default — will raise AttributeError if missing",
                        source_line=line,
                    )
                )

    return issues


def _is_inside_try(node: ast.AST) -> bool:
    """Check if a node is inside a try/except block."""
    current = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if isinstance(current, ast.Try):
            return True
    return False


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    files = collect_files()
    all_issues: list[Issue] = []
    stats = {"files": 0, "calls_checked": 0, "calls_skipped": 0}

    for fp in files:
        try:
            source = fp.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(fp))
        except (SyntaxError, UnicodeDecodeError):
            continue

        source_lines = source.splitlines()
        _annotate_parents(tree)
        stats["files"] += 1

        rel = str(fp.relative_to(PROJECT_ROOT))

        # --- Signature checks ---
        imports = parse_imports(tree)
        import_map = {imp.local_name: imp for imp in imports}
        calls = collect_calls(tree, set(import_map.keys()), rel, source_lines)

        for call in calls:
            imp = import_map.get(call.func_name)
            if not imp:
                stats["calls_skipped"] += 1
                continue

            if call.has_starargs or call.has_starkwargs:
                stats["calls_skipped"] += 1
                continue

            sig, is_class = resolve_signature(imp)
            if sig is None:
                stats["calls_skipped"] += 1
                continue

            stats["calls_checked"] += 1
            fqn = f"{imp.module_path}.{imp.attr_name}"
            all_issues.extend(check_signature(call, sig, is_class, fqn))

        # --- Annotation checks ---
        all_issues.extend(check_annotations(tree, rel, source_lines))

        # --- Unsafe access checks ---
        all_issues.extend(check_unsafe_access(tree, rel, source_lines))

    # --- Report ---
    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]

    print()
    print("Signature & Type Verification Report")
    print("=" * 60)
    print(f"Files scanned:    {stats['files']}")
    print(f"Calls checked:    {stats['calls_checked']}")
    print(f"Calls skipped:    {stats['calls_skipped']}")
    print(f"Errors:           {len(errors)}")
    print(f"Warnings:         {len(warnings)}")
    print()

    if errors:
        print("ERRORS (must fix)")
        print("-" * 60)
        for issue in sorted(errors, key=lambda x: (x.file_path, x.lineno)):
            print(f"  {issue.file_path}:{issue.lineno} [{issue.category}]")
            print(f"    {issue.source_line}")
            print(f"    {issue.message}")
            print()

    if warnings:
        # Group by category for readability
        by_cat: dict[str, list[Issue]] = {}
        for w in warnings:
            by_cat.setdefault(w.category, []).append(w)

        print("WARNINGS")
        print("-" * 60)
        for cat, items in sorted(by_cat.items()):
            print(f"  [{cat}] ({len(items)} issues)")
            for issue in sorted(items, key=lambda x: (x.file_path, x.lineno))[:10]:
                print(f"    {issue.file_path}:{issue.lineno}")
                print(f"      {issue.message}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")
            print()

    if not errors and not warnings:
        print("  No issues found.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
