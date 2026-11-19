# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

FROM python:3.11-slim
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY config ./config
COPY fixtures ./fixtures
ENV SIGNALGRAPH_DB=/data/signalgraph.duckdb \
    SIGNALGRAPH_CONFIG_DIR=/app/config \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
ENTRYPOINT ["signalgraph"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
