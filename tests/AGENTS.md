# tests/ AGENTS

Unit tests (pytest runner; tests written in unittest style).

## Overview
- Test discovery is configured in `pyproject.toml` (`tests/`, `test_*.py`).
- `tests/__init__.py` appends repo root to `sys.path` for imports.

## Commands
```bash
pytest tests
```

## Conventions / gotchas
- Some tests intentionally skip optional dependencies (broad `skipTest(...)` patterns).
- Prefer lightweight synthetic tensors; avoid requiring datasets.
