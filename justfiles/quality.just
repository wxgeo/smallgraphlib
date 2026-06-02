ruff:
    {{uv}} run ruff check src tests

mypy:
    {{uv}} run mypy src tests

pytest:
    {{uv}} run pytest -n auto tests

test: (ruff) (mypy) (pytest)

fix:
    {{uv}} run ruff format src tests
    {{uv}} run ruff check --fix src tests
