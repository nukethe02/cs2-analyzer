# Contributing to OpenSight

Thank you for your interest in contributing to OpenSight! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, constructive, and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A CS2 demo file for testing (optional but helpful)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/opensight.git
   cd opensight
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit install --hook-type commit-msg
   ```

5. **Verify your setup**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names following this pattern:
- `feature/` - New features (e.g., `feature/add-heatmap-export`)
- `fix/` - Bug fixes (e.g., `fix/ttd-calculation-edge-case`)
- `docs/` - Documentation updates (e.g., `docs/improve-api-examples`)
- `refactor/` - Code refactoring (e.g., `refactor/metrics-module`)
- `test/` - Test additions/improvements (e.g., `test/parser-edge-cases`)

### Making Changes

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines (enforced by pre-commit)
   - Write tests for new functionality
   - Update documentation as needed

3. **Run tests locally**
   ```bash
   pytest tests/ -v
   ```

4. **Run linting**
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add heatmap export functionality"
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only changes
- `style` - Code style changes (formatting, semicolons, etc.)
- `refactor` - Code refactoring without feature changes
- `perf` - Performance improvements
- `test` - Adding or updating tests
- `build` - Changes to build system or dependencies
- `ci` - Changes to CI configuration
- `chore` - Other changes that don't modify src or test files

**Examples:**
```
feat(metrics): add utility usage tracking
fix(parser): handle missing view angle data
docs(readme): add installation instructions
test(watcher): add integration tests for file detection
```

## Code Style

### Python Style Guide

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and under 50 lines when possible
- Use descriptive variable names

### Example Function

```python
def calculate_damage_per_round(
    damage_events: pd.DataFrame,
    round_starts: list[int],
    player_id: int,
) -> float:
    """
    Calculate average damage per round for a player.

    Args:
        damage_events: DataFrame containing damage event data
        round_starts: List of ticks where rounds start
        player_id: Steam ID of the player to analyze

    Returns:
        Average damage dealt per round

    Raises:
        ValueError: If no damage events are found
    """
    player_damage = damage_events[damage_events["attacker_id"] == player_id]

    if player_damage.empty:
        return 0.0

    total_damage = player_damage["damage"].sum()
    num_rounds = max(len(round_starts), 1)

    return total_damage / num_rounds
```

### Testing Guidelines

- Write tests for all new functionality
- Use pytest fixtures for common test data
- Test edge cases and error conditions
- Aim for meaningful coverage, not just high percentages
- Use descriptive test names that explain what's being tested

```python
class TestDamagePerRound:
    """Tests for damage per round calculation."""

    def test_returns_zero_for_no_damage(self, empty_demo_data):
        """Verify zero is returned when player dealt no damage."""
        result = calculate_damage_per_round(
            empty_demo_data.damage_events,
            empty_demo_data.round_starts,
            player_id=12345,
        )
        assert result == 0.0

    def test_calculates_average_correctly(self, sample_demo_data):
        """Verify average is calculated correctly over multiple rounds."""
        # Player dealt 300 damage over 3 rounds = 100 DPR
        result = calculate_damage_per_round(
            sample_demo_data.damage_events,
            sample_demo_data.round_starts,
            player_id=1,
        )
        assert result == pytest.approx(100.0, abs=0.1)
```

## Project Structure

```
opensight/
├── src/opensight/          # Main package
│   ├── __init__.py         # Package exports
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── export.py           # Export functionality
│   ├── metrics.py          # Metrics calculations
│   ├── parser.py           # Demo parsing
│   ├── sharecode.py        # Share code encoding/decoding
│   └── watcher.py          # Replay file monitoring
├── tests/                  # Test suite
│   ├── test_metrics.py
│   ├── test_parser.py
│   ├── test_sharecode.py
│   └── test_watcher.py
├── pyproject.toml          # Project configuration
├── README.md               # Project documentation
└── CONTRIBUTING.md         # This file
```

## Adding New Features

### New Metrics

1. Add the dataclass in `src/opensight/metrics.py`
2. Implement the calculation function
3. Add tests in `tests/test_metrics.py`
4. Update `ComprehensivePlayerMetrics` if applicable
5. Add CLI support if needed
6. Update documentation

### New Export Formats

1. Add the export function in `src/opensight/export.py`
2. Update `export_analysis()` to support the new format
3. Add tests
4. Update CLI if needed

### New CLI Commands

1. Add the command in `src/opensight/cli.py`
2. Follow the existing pattern with Typer
3. Add appropriate options and help text
4. Test the command manually

## Submitting Changes

### Pull Request Process

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**
   - Use a clear, descriptive title
   - Fill out the PR template
   - Link any related issues

3. **PR Description Template**
   ```markdown
   ## Summary
   Brief description of the changes.

   ## Changes
   - Added X
   - Fixed Y
   - Updated Z

   ## Testing
   - [ ] Tests pass locally
   - [ ] New tests added for new functionality
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Commit messages follow conventions
   ```

4. **Review Process**
   - Address reviewer feedback promptly
   - Push additional commits to address feedback
   - Squash commits if requested

### Review Criteria

PRs are evaluated on:
- Code quality and style
- Test coverage
- Documentation
- Backwards compatibility
- Performance impact

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)
- Demo file characteristics (if relevant)

### Feature Requests

Include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach (if known)
- Examples from other tools (if applicable)

## Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions
- Check existing issues and PRs before creating new ones

## Recognition

Contributors are recognized in:
- The project's AUTHORS file
- Release notes for significant contributions
- GitHub's contributors graph

Thank you for contributing to OpenSight!
