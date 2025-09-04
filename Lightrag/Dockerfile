# ---- Stage 1: Builder ----
# Use a specific slim version for reproducibility
FROM python:3.11.9-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Install build dependencies for Rust compilation and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable

# Copy only requirements files first to leverage Docker layer caching
COPY requirements.txt .
RUN mkdir -p ./lightrag/api
COPY lightrag/api/requirements.txt ./lightrag/api/

RUN echo "wcwidth" >> requirements.txt

# Install Python dependencies into the user site-packages directory within the builder stage
# Using --user in the builder means they install to /root/.local
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user -r lightrag/api/requirements.txt

# Install dependencies for default storage
RUN pip install --user --no-cache-dir nano-vectordb networkx
# Install dependencies for default LLM
RUN pip install --user --no-cache-dir openai ollama tiktoken
# Install dependencies for default document loader
RUN pip install --user --no-cache-dir pypdf2 python-docx python-pptx openpyxl

# Copy the rest of the application code needed for installation
COPY ./lightrag ./lightrag
COPY setup.py .

# Install the LightRAG package itself into the user site-packages
# This makes the actual 'lightrag' library available
RUN pip install --no-cache-dir --user .

# ---- Stage 2: Final Image ----
# Use the same base image for consistency
FROM python:3.11.9-slim-bookworm

# Set working directory
WORKDIR /app

# Define user and group IDs
ARG APP_UID=1001
ARG APP_GID=1001

# Create a non-root user and group with specified IDs
RUN groupadd --gid ${APP_GID} lightrag && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home lightrag

# Copy installed packages from the builder stage user directory
# These will initially be owned by root after copying
COPY --from=builder /root/.local /home/lightrag/.local

# --- IMPORTANT: Fix ownership of the copied .local directory ---
# Ensure the 'lightrag' user owns its site-packages directory
RUN chown -R lightrag:lightrag /home/lightrag/.local

# Copy application code (already installed, but might be needed for assets or runtime access)
# Set ownership during copy
COPY --chown=lightrag:lightrag ./lightrag ./lightrag
# setup.py likely not needed in final image, as installation happened in builder stage
# COPY --chown=lightrag:lightrag setup.py . # Vous pouvez commenter/supprimer ceci

# --- FIX: Explicitly grant read/execute permissions ---
# Ensure the user/group can read the code and traverse directories
RUN chmod -R ug+rX /app/lightrag 

# Set PATH to include the user's local bin directory and system bin
ENV PATH=/home/lightrag/.local/bin:/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # Set data directories expected by the application (can be overridden by K8s env)
    WORKING_DIR=/app/data/rag_storage \
    INPUT_DIR=/app/data/inputs

# Create application data directories and set ownership of /app and /app/data
RUN mkdir -p ${INPUT_DIR} ${WORKING_DIR} && \
    chown -R lightrag:lightrag /app/data /app

# Switch to the non-root user
USER lightrag

# Expose the default port the application listens on
EXPOSE 9621

# Set the default command to run for the container
# Use the production Gunicorn script. Arguments (--host, --port) will be provided by Kubernetes.
ENTRYPOINT ["python", "-m", "lightrag.api.run_with_gunicorn"]