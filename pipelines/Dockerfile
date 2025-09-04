FROM ghcr.io/open-webui/pipelines:main

# Installer dependencies depuis GitHub + langchain_openai/ollama + psycopg
RUN pip install --no-cache-dir git+https://github.com/Ryustiel/Raphlib.git \
    && pip install --no-cache-dir git+https://github.com/Ryustiel/Ragraph.git \
    && pip install --no-cache-dir git+https://github.com/Ryustiel/clients.git \
    && pip install --no-cache-dir langchain_openai \
    && pip install --no-cache-dir langchain_ollama \
    && pip install --no-cache-dir psycopg[binary]