# Base: Dev Containers Python 3.11 on Ubuntu
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install build tools (for compiling native deps when needed)
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for the vscode user
USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
ENV RUSTUP_HOME=/home/vscode/.rustup
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y && \
    /home/vscode/.cargo/bin/rustup default stable
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

# Keep default user as vscode
USER vscode
