# Contributing

ForecastFM optimizes for code that can be read during an audit.

- Prefer a function over a class until state or multiple implementations require a class.
- Prefer explicit data flow over callbacks, registries, decorators, and dependency injection.
- Keep domain logic independent from Tinker and other vendors.
- Validate data at the boundary, then use precise types internally.
- Use timezone-aware UTC timestamps for every point-in-time field.
- Add a test for every bug and every validation rule.
- Do not add a dependency for something the standard library expresses clearly.
- Do not weaken Pyright or Ruff rules to merge code. Fix the code or document a narrow exception.

Before committing, run:

```bash
uv run --extra tinker ruff format --check .
uv run --extra tinker ruff check .
uv run --extra tinker pyright
uv run --extra tinker pytest
```
