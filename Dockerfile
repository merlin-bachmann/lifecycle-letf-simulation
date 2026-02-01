FROM python:3.14.0-slim as builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

RUN pip install uv

WORKDIR /app
COPY pyproject.toml .
COPY uv.lock .

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM python:3.14.0-slim as runtime

COPY --from=builder --chown=app:app /app /app

ENV VIRTUAL_ENV=/app/.venv PATH="/app/.venv/bin:$PATH"

COPY src src
# COPY static static # Uncomment this line if you have static assets.

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
