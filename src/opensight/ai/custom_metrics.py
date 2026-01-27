"""
Custom Metric Builder Module for CS2 Demo Analyzer.

Allows users to define and track custom metrics using a formula editor.
Supports mathematical expressions, aggregations, and conditional logic.
"""

import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Safe Expression Evaluator
# ============================================================================


class SafeExpressionEvaluator:
    """
    Safely evaluates mathematical expressions without using eval().
    Supports basic math, aggregations, and conditionals.
    """

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Comparison operators
    COMPARISONS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    # Boolean operators
    BOOL_OPS = {
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
    }

    # Built-in math functions
    FUNCTIONS = {
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sqrt": math.sqrt,
        "pow": pow,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "sum": sum,
        "avg": lambda x: sum(x) / len(x) if x else 0,
        "mean": lambda x: sum(x) / len(x) if x else 0,
        "median": lambda x: sorted(x)[len(x) // 2] if x else 0,
        "count": len,
        "len": len,
    }

    # Constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        self.variables: dict[str, Any] = {}

    def evaluate(self, expression: str, variables: dict[str, Any] = None) -> Any:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string
            variables: Variable values to substitute

        Returns:
            Evaluated result

        Raises:
            ValueError: If expression is invalid or uses disallowed operations
        """
        if variables:
            self.variables = variables
        else:
            self.variables = {}

        try:
            tree = ast.parse(expression, mode="eval")
            return self._eval_node(tree.body)
        except (SyntaxError, ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid expression: {e}") from e

    def _eval_node(self, node: ast.expr) -> Any:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n

        elif isinstance(node, ast.Str):  # Python 3.7 compatibility
            return node.s

        elif isinstance(node, ast.Name):
            name = node.id
            if name in self.CONSTANTS:
                return self.CONSTANTS[name]
            if name in self.variables:
                return self.variables[name]
            raise ValueError(f"Unknown variable: {name}")

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._eval_node(comparator)
                comp_op = self.COMPARISONS.get(type(op))
                if comp_op is None:
                    raise ValueError(f"Unsupported comparison: {type(op).__name__}")
                if not comp_op(left, right):
                    return False
                left = right
            return True

        elif isinstance(node, ast.BoolOp):
            op = self.BOOL_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
            result = self._eval_node(node.values[0])
            for value in node.values[1:]:
                result = op(result, self._eval_node(value))
            return result

        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test)
            if test:
                return self._eval_node(node.body)
            else:
                return self._eval_node(node.orelse)

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]

            # Handle variadic functions
            func = self.FUNCTIONS[func_name]
            if func_name in ["min", "max", "sum", "avg", "mean", "median", "count", "len"]:
                # If single list argument, use it directly
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return func(args[0])
                return func(args)
            else:
                return func(*args)

        elif isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)

        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            if isinstance(node.slice, ast.Index):  # Python 3.8 compatibility
                index = self._eval_node(node.slice.value)
            else:
                index = self._eval_node(node.slice)
            return value[index]

        elif isinstance(node, ast.Attribute):
            value = self._eval_node(node.value)
            if isinstance(value, dict):
                return value.get(node.attr, 0)
            return getattr(value, node.attr, 0)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# ============================================================================
# Custom Metric Definitions
# ============================================================================


class MetricType(Enum):
    """Types of custom metrics."""

    SCALAR = "scalar"  # Single value
    PERCENTAGE = "percentage"  # 0-100%
    RATIO = "ratio"  # Decimal ratio
    TIME_MS = "time_ms"  # Time in milliseconds
    COUNT = "count"  # Integer count
    RATE = "rate"  # Per-round rate
    COMPOUND = "compound"  # Complex multi-step metric


class AggregationType(Enum):
    """How to aggregate metric over multiple rounds/demos."""

    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"


@dataclass
class MetricVariable:
    """A variable that can be used in custom metric formulas."""

    name: str
    description: str
    data_path: str  # Path to extract from stats (e.g., "kills", "utility_stats.flashbangs_thrown")
    default_value: Any = 0
    transform: str | None = None  # Optional transform formula

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "data_path": self.data_path,
            "default_value": self.default_value,
            "transform": self.transform,
        }


@dataclass
class CustomMetric:
    """A user-defined custom metric."""

    metric_id: str
    name: str
    description: str
    formula: str
    metric_type: MetricType = MetricType.SCALAR
    aggregation: AggregationType = AggregationType.AVERAGE

    # Formula validation
    variables_used: list[str] = field(default_factory=list)
    is_valid: bool = True
    validation_error: str | None = None

    # Display settings
    format_string: str = "{:.2f}"
    unit: str = ""
    higher_is_better: bool = True

    # Benchmarks
    benchmark_low: float | None = None
    benchmark_mid: float | None = None
    benchmark_high: float | None = None

    # Metadata
    created_at: str = ""
    created_by: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "metric_type": self.metric_type.value,
            "aggregation": self.aggregation.value,
            "variables_used": self.variables_used,
            "is_valid": self.is_valid,
            "validation_error": self.validation_error,
            "display": {
                "format": self.format_string,
                "unit": self.unit,
                "higher_is_better": self.higher_is_better,
            },
            "benchmarks": {
                "low": self.benchmark_low,
                "mid": self.benchmark_mid,
                "high": self.benchmark_high,
            },
            "created_at": self.created_at,
            "created_by": self.created_by,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomMetric":
        display = data.get("display", {})
        benchmarks = data.get("benchmarks", {})

        return cls(
            metric_id=data.get("metric_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            formula=data.get("formula", ""),
            metric_type=MetricType(data.get("metric_type", "scalar")),
            aggregation=AggregationType(data.get("aggregation", "average")),
            variables_used=data.get("variables_used", []),
            is_valid=data.get("is_valid", True),
            validation_error=data.get("validation_error"),
            format_string=display.get("format", "{:.2f}"),
            unit=display.get("unit", ""),
            higher_is_better=display.get("higher_is_better", True),
            benchmark_low=benchmarks.get("low"),
            benchmark_mid=benchmarks.get("mid"),
            benchmark_high=benchmarks.get("high"),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", ""),
            tags=data.get("tags", []),
        )


@dataclass
class MetricResult:
    """Result of calculating a custom metric."""

    metric_id: str
    metric_name: str
    value: float
    formatted_value: str
    rating: str  # "low", "average", "good", "excellent"
    per_round_values: list[float] = field(default_factory=list)
    calculation_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "rating": self.rating,
            "per_round_values": self.per_round_values,
            "calculation_details": self.calculation_details,
        }


# ============================================================================
# Built-in Variable Library
# ============================================================================

BUILTIN_VARIABLES = [
    # Basic stats
    MetricVariable("kills", "Total kills", "kills"),
    MetricVariable("deaths", "Total deaths", "deaths"),
    MetricVariable("assists", "Total assists", "assists"),
    MetricVariable("rounds", "Rounds played", "rounds_played", default_value=1),
    MetricVariable("headshots", "Headshot kills", "headshots"),
    MetricVariable("damage", "Total damage dealt", "total_damage"),
    # Derived stats
    MetricVariable("kd", "Kill/Death ratio", "kd_ratio"),
    MetricVariable("adr", "Average damage per round", "adr"),
    MetricVariable("hs_percent", "Headshot percentage", "hs_percentage"),
    MetricVariable("kast", "KAST percentage", "kast_percentage"),
    MetricVariable("hltv", "HLTV rating", "hltv_rating"),
    MetricVariable("impact", "Impact rating", "impact_rating"),
    # Aim metrics
    MetricVariable("ttd", "Time to damage (ms)", "ttd_median_ms"),
    MetricVariable("cp_error", "Crosshair placement error (deg)", "cp_median_error"),
    MetricVariable("accuracy", "Accuracy percentage", "accuracy"),
    MetricVariable("prefire_count", "Prefire kills", "prefire_count"),
    # Opening duels
    MetricVariable("opening_wins", "Opening duel wins", "opening_duel_stats.wins"),
    MetricVariable("opening_losses", "Opening duel losses", "opening_duel_stats.losses"),
    MetricVariable("opening_attempts", "Opening duel attempts", "opening_duel_stats.attempts"),
    # Trades
    MetricVariable("trades_given", "Trade kills given", "trade_stats.kills_traded"),
    MetricVariable("trades_received", "Trade deaths received", "trade_stats.deaths_traded"),
    # Clutches
    MetricVariable("clutch_wins", "Clutch rounds won", "clutch_stats.total_wins"),
    MetricVariable("clutch_attempts", "Clutch attempts", "clutch_stats.total_situations"),
    # Multi-kills
    MetricVariable("rounds_2k", "Rounds with 2+ kills", "multi_kill_stats.rounds_with_2k"),
    MetricVariable("rounds_3k", "Rounds with 3+ kills", "multi_kill_stats.rounds_with_3k"),
    MetricVariable("rounds_4k", "Rounds with 4+ kills", "multi_kill_stats.rounds_with_4k"),
    MetricVariable("rounds_5k", "Ace rounds", "multi_kill_stats.rounds_with_5k"),
    # Utility
    MetricVariable("flashes", "Flashbangs thrown", "utility_stats.flashbangs_thrown"),
    MetricVariable("smokes", "Smokes thrown", "utility_stats.smokes_thrown"),
    MetricVariable("he_nades", "HE grenades thrown", "utility_stats.he_thrown"),
    MetricVariable("molotovs", "Molotovs thrown", "utility_stats.molotovs_thrown"),
    MetricVariable("enemies_flashed", "Enemies flashed", "utility_stats.enemies_flashed"),
    MetricVariable("teammates_flashed", "Teammates flashed", "utility_stats.teammates_flashed"),
    MetricVariable("he_damage", "HE grenade damage", "utility_stats.he_damage"),
    MetricVariable("molotov_damage", "Molotov damage", "utility_stats.molotov_damage"),
    # Economy
    MetricVariable("equipment_value", "Equipment value", "economy.equipment_value"),
    MetricVariable("kills_per_dollar", "Kills per dollar", "economy.kills_per_dollar"),
    # Mistakes
    MetricVariable("team_kills", "Team kills", "mistakes_stats.team_kills"),
    MetricVariable("team_damage", "Team damage", "mistakes_stats.team_damage"),
    # Side-specific
    MetricVariable("ct_kills", "CT side kills", "ct_stats.kills"),
    MetricVariable("ct_deaths", "CT side deaths", "ct_stats.deaths"),
    MetricVariable("t_kills", "T side kills", "t_stats.kills"),
    MetricVariable("t_deaths", "T side deaths", "t_stats.deaths"),
]


# ============================================================================
# Custom Metric Builder
# ============================================================================


class CustomMetricBuilder:
    """
    Builds and validates custom metrics from user-defined formulas.
    """

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or Path.home() / ".opensight" / "custom_metrics"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = SafeExpressionEvaluator()
        self.variables = {v.name: v for v in BUILTIN_VARIABLES}
        self.metrics: dict[str, CustomMetric] = {}

        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load saved custom metrics from disk."""
        metrics_file = self.data_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                for m_data in data.get("metrics", []):
                    metric = CustomMetric.from_dict(m_data)
                    self.metrics[metric.metric_id] = metric
            except (OSError, json.JSONDecodeError):
                pass

    def _save_metrics(self) -> None:
        """Save custom metrics to disk."""
        metrics_file = self.data_dir / "metrics.json"
        try:
            with open(metrics_file, "w") as f:
                json.dump({"metrics": [m.to_dict() for m in self.metrics.values()]}, f, indent=2)
        except OSError:
            pass

    def create_metric(
        self,
        name: str,
        formula: str,
        description: str = "",
        metric_type: MetricType = MetricType.SCALAR,
        aggregation: AggregationType = AggregationType.AVERAGE,
        unit: str = "",
        higher_is_better: bool = True,
        benchmarks: tuple[float, float, float] | None = None,
        created_by: str = "",
    ) -> CustomMetric:
        """
        Create a new custom metric.

        Args:
            name: Metric display name
            formula: Mathematical formula using available variables
            description: Human-readable description
            metric_type: Type of metric value
            aggregation: How to aggregate across rounds
            unit: Display unit (e.g., "ms", "%")
            higher_is_better: Whether higher values are better
            benchmarks: Optional (low, mid, high) benchmark values
            created_by: Creator identifier

        Returns:
            Created CustomMetric object
        """
        import hashlib

        # Generate ID
        metric_id = hashlib.md5(
            f"{name}_{formula}_{datetime.now().isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        # Validate formula
        is_valid, error, variables_used = self._validate_formula(formula)

        metric = CustomMetric(
            metric_id=metric_id,
            name=name,
            description=description,
            formula=formula,
            metric_type=metric_type,
            aggregation=aggregation,
            variables_used=variables_used,
            is_valid=is_valid,
            validation_error=error,
            unit=unit,
            higher_is_better=higher_is_better,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
        )

        if benchmarks:
            metric.benchmark_low = benchmarks[0]
            metric.benchmark_mid = benchmarks[1]
            metric.benchmark_high = benchmarks[2]

        # Set format string based on type
        if metric_type == MetricType.PERCENTAGE:
            metric.format_string = "{:.1f}%"
        elif metric_type == MetricType.TIME_MS:
            metric.format_string = "{:.0f}ms"
        elif metric_type == MetricType.COUNT:
            metric.format_string = "{:.0f}"
        elif metric_type == MetricType.RATIO:
            metric.format_string = "{:.2f}"

        self.metrics[metric_id] = metric
        self._save_metrics()

        return metric

    def _validate_formula(self, formula: str) -> tuple[bool, str | None, list[str]]:
        """
        Validate a metric formula.

        Returns:
            Tuple of (is_valid, error_message, variables_used)
        """
        variables_used = []

        # Find all variable references
        # Match word characters that aren't function names
        func_names = set(SafeExpressionEvaluator.FUNCTIONS.keys())
        const_names = set(SafeExpressionEvaluator.CONSTANTS.keys())

        # Extract potential variable names
        potential_vars = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", formula)

        for var in potential_vars:
            if var not in func_names and var not in const_names:
                if var not in self.variables:
                    return False, f"Unknown variable: {var}", []
                variables_used.append(var)

        # Try to parse the formula
        try:
            ast.parse(formula, mode="eval")
        except SyntaxError as e:
            return False, f"Syntax error: {e}", variables_used

        # Test evaluation with dummy values
        try:
            test_vars = dict.fromkeys(variables_used, 1.0)
            self.evaluator.evaluate(formula, test_vars)
        except Exception as e:
            return False, f"Evaluation error: {e}", variables_used

        return True, None, list(set(variables_used))

    def update_metric(self, metric_id: str, **kwargs) -> CustomMetric | None:
        """Update an existing metric."""
        if metric_id not in self.metrics:
            return None

        metric = self.metrics[metric_id]

        for key, value in kwargs.items():
            if hasattr(metric, key):
                setattr(metric, key, value)

        # Re-validate if formula changed
        if "formula" in kwargs:
            is_valid, error, variables_used = self._validate_formula(metric.formula)
            metric.is_valid = is_valid
            metric.validation_error = error
            metric.variables_used = variables_used

        self._save_metrics()
        return metric

    def delete_metric(self, metric_id: str) -> bool:
        """Delete a custom metric."""
        if metric_id in self.metrics:
            del self.metrics[metric_id]
            self._save_metrics()
            return True
        return False

    def get_metric(self, metric_id: str) -> CustomMetric | None:
        """Get a metric by ID."""
        return self.metrics.get(metric_id)

    def list_metrics(self) -> list[CustomMetric]:
        """List all custom metrics."""
        return list(self.metrics.values())

    def list_variables(self) -> list[dict[str, Any]]:
        """List all available variables."""
        return [v.to_dict() for v in self.variables.values()]


# ============================================================================
# Metric Calculator
# ============================================================================


class MetricCalculator:
    """
    Calculates custom metric values from player statistics.
    """

    def __init__(self, builder: CustomMetricBuilder):
        self.builder = builder
        self.evaluator = SafeExpressionEvaluator()

    def calculate(self, metric: CustomMetric, player_stats: dict[str, Any]) -> MetricResult:
        """
        Calculate a custom metric for a player.

        Args:
            metric: Custom metric to calculate
            player_stats: Player statistics dictionary

        Returns:
            MetricResult with calculated value
        """
        if not metric.is_valid:
            return MetricResult(
                metric_id=metric.metric_id,
                metric_name=metric.name,
                value=0,
                formatted_value="Error",
                rating="error",
                calculation_details={"error": metric.validation_error},
            )

        # Extract variable values
        variables = {}
        for var_name in metric.variables_used:
            var_def = self.builder.variables.get(var_name)
            if var_def:
                value = self._extract_value(player_stats, var_def.data_path, var_def.default_value)
                variables[var_name] = value

        # Evaluate formula
        try:
            value = self.evaluator.evaluate(metric.formula, variables)
        except Exception as e:
            return MetricResult(
                metric_id=metric.metric_id,
                metric_name=metric.name,
                value=0,
                formatted_value="Error",
                rating="error",
                calculation_details={"error": str(e), "variables": variables},
            )

        # Format value
        try:
            formatted = metric.format_string.format(value)
        except (ValueError, KeyError):
            formatted = str(round(value, 2))

        if metric.unit and not metric.format_string.endswith(metric.unit):
            formatted = f"{formatted}{metric.unit}"

        # Determine rating
        rating = self._determine_rating(value, metric)

        return MetricResult(
            metric_id=metric.metric_id,
            metric_name=metric.name,
            value=value,
            formatted_value=formatted,
            rating=rating,
            calculation_details={"variables": variables, "formula": metric.formula},
        )

    def calculate_all(self, player_stats: dict[str, Any]) -> list[MetricResult]:
        """Calculate all custom metrics for a player."""
        results = []
        for metric in self.builder.list_metrics():
            result = self.calculate(metric, player_stats)
            results.append(result)
        return results

    def _extract_value(self, stats: dict[str, Any], path: str, default: Any) -> Any:
        """Extract a value from nested dictionary using dot notation."""
        parts = path.split(".")
        value = stats

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part, default)
            elif hasattr(value, part):
                value = getattr(value, part, default)
            else:
                return default

        return value if value is not None else default

    def _determine_rating(self, value: float, metric: CustomMetric) -> str:
        """Determine performance rating based on benchmarks."""
        if metric.benchmark_low is None or metric.benchmark_high is None:
            return "average"

        if metric.higher_is_better:
            if value >= metric.benchmark_high:
                return "excellent"
            elif metric.benchmark_mid and value >= metric.benchmark_mid:
                return "good"
            elif value >= metric.benchmark_low:
                return "average"
            else:
                return "low"
        else:
            if value <= metric.benchmark_low:
                return "excellent"
            elif metric.benchmark_mid and value <= metric.benchmark_mid:
                return "good"
            elif value <= metric.benchmark_high:
                return "average"
            else:
                return "low"


# ============================================================================
# Preset Metrics Library
# ============================================================================

PRESET_METRICS = [
    {
        "name": "Time to Site",
        "formula": "ttd + cp_error * 10",
        "description": "Combined measure of reaction time and crosshair placement",
        "metric_type": "time_ms",
        "unit": "ms",
        "higher_is_better": False,
        "benchmarks": (200, 350, 500),
    },
    {
        "name": "Entry Impact",
        "formula": "(opening_wins * 2 - opening_losses) / max(opening_attempts, 1)",
        "description": "Impact of opening duel performance",
        "metric_type": "ratio",
        "higher_is_better": True,
        "benchmarks": (-0.5, 0.2, 0.8),
    },
    {
        "name": "Utility Efficiency",
        "formula": "(enemies_flashed - teammates_flashed + he_damage / 25 + molotov_damage / 20) / max(flashes + he_nades + molotovs, 1)",
        "description": "Effectiveness of utility usage per grenade",
        "metric_type": "ratio",
        "higher_is_better": True,
        "benchmarks": (0.3, 0.7, 1.2),
    },
    {
        "name": "Clutch Factor",
        "formula": "clutch_wins * 100 / max(clutch_attempts, 1)",
        "description": "Clutch round win percentage",
        "metric_type": "percentage",
        "unit": "%",
        "higher_is_better": True,
        "benchmarks": (20, 35, 50),
    },
    {
        "name": "Multi-Kill Rate",
        "formula": "(rounds_2k + rounds_3k * 2 + rounds_4k * 3 + rounds_5k * 5) / rounds * 100",
        "description": "Weighted multi-kill frequency",
        "metric_type": "percentage",
        "unit": "%",
        "higher_is_better": True,
        "benchmarks": (10, 25, 45),
    },
    {
        "name": "Trade Balance",
        "formula": "trades_given - trades_received",
        "description": "Net trade balance (positive = giving more trades)",
        "metric_type": "count",
        "higher_is_better": True,
        "benchmarks": (-2, 1, 4),
    },
    {
        "name": "Side Difference",
        "formula": "(ct_kills - ct_deaths) - (t_kills - t_deaths)",
        "description": "Performance difference between CT and T sides",
        "metric_type": "count",
        "higher_is_better": False,  # Closer to 0 is better (balanced)
        "benchmarks": (-3, 0, 3),
    },
    {
        "name": "Mistake Index",
        "formula": "team_kills * 10 + team_damage / 10 + teammates_flashed * 2",
        "description": "Weighted sum of friendly fire incidents",
        "metric_type": "scalar",
        "higher_is_better": False,
        "benchmarks": (5, 15, 30),
    },
    {
        "name": "Eco Efficiency",
        "formula": "damage / max(equipment_value / 1000, 1)",
        "description": "Damage dealt per $1000 spent",
        "metric_type": "ratio",
        "higher_is_better": True,
        "benchmarks": (20, 40, 70),
    },
    {
        "name": "Consistency Score",
        "formula": "kast * 0.4 + (1 - abs(hltv - 1) * 2) * 30 + accuracy * 0.3",
        "description": "Combined consistency rating",
        "metric_type": "scalar",
        "higher_is_better": True,
        "benchmarks": (40, 55, 70),
    },
]


def create_preset_metrics(builder: CustomMetricBuilder) -> list[CustomMetric]:
    """Create all preset metrics."""
    created = []
    for preset in PRESET_METRICS:
        try:
            metric = builder.create_metric(
                name=preset["name"],
                formula=preset["formula"],
                description=preset["description"],
                metric_type=MetricType(preset["metric_type"]),
                unit=preset.get("unit", ""),
                higher_is_better=preset["higher_is_better"],
                benchmarks=preset.get("benchmarks"),
                created_by="system",
            )
            created.append(metric)
        except Exception:
            continue
    return created


# ============================================================================
# Convenience Functions
# ============================================================================

_default_builder: CustomMetricBuilder | None = None


def get_builder() -> CustomMetricBuilder:
    """Get or create the default metric builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = CustomMetricBuilder()
    return _default_builder


def create_custom_metric(name: str, formula: str, **kwargs) -> dict[str, Any]:
    """
    Create a custom metric.

    Args:
        name: Metric name
        formula: Mathematical formula
        **kwargs: Additional metric options

    Returns:
        Created metric dictionary
    """
    builder = get_builder()
    metric = builder.create_metric(name, formula, **kwargs)
    return metric.to_dict()


def calculate_custom_metrics(player_stats: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Calculate all custom metrics for a player.

    Args:
        player_stats: Player statistics

    Returns:
        List of metric results
    """
    builder = get_builder()
    calculator = MetricCalculator(builder)
    results = calculator.calculate_all(player_stats)
    return [r.to_dict() for r in results]


def calculate_metric(metric_id: str, player_stats: dict[str, Any]) -> dict[str, Any]:
    """Calculate a specific custom metric."""
    builder = get_builder()
    metric = builder.get_metric(metric_id)
    if not metric:
        return {"error": "Metric not found"}

    calculator = MetricCalculator(builder)
    result = calculator.calculate(metric, player_stats)
    return result.to_dict()


def list_available_variables() -> list[dict[str, Any]]:
    """List all variables available for formulas."""
    return get_builder().list_variables()


def list_custom_metrics() -> list[dict[str, Any]]:
    """List all custom metrics."""
    return [m.to_dict() for m in get_builder().list_metrics()]


def validate_formula(formula: str) -> dict[str, Any]:
    """
    Validate a formula without creating a metric.

    Args:
        formula: Formula to validate

    Returns:
        Validation result
    """
    builder = get_builder()
    is_valid, error, variables = builder._validate_formula(formula)
    return {"valid": is_valid, "error": error, "variables_used": variables}


def install_preset_metrics() -> list[dict[str, Any]]:
    """Install all preset metrics."""
    builder = get_builder()
    metrics = create_preset_metrics(builder)
    return [m.to_dict() for m in metrics]
