FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/assets/generated

EXPOSE 8080

ENTRYPOINT ["python", "-m", "asset_forge_mcp.server"]
