# Build arguments
ARG RUST_VERSION=latest
ARG DEBIAN_VERSION=bookworm

# Use cargo-chef for better caching
FROM lukemathwalker/cargo-chef:${RUST_VERSION}-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Build dependencies
FROM chef AS builder
RUN apt-get update -y && apt-get install -y clang && rm -rf /var/lib/apt/lists/*
COPY --from=planner /app/recipe.json recipe.json

# Build dependencies - cached if they don't change
RUN cargo chef cook --release --recipe-path recipe.json

# Build application
COPY . .
RUN cargo build --release -p ferrules-api

# Runtime stage
FROM debian:${DEBIAN_VERSION}-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary and libs from builder
COPY --from=builder /app/target/release/libonnxruntime*.so /usr/local/lib/
COPY --from=builder /app/target/release/ferrules-api /app/ferrules-api

RUN ldconfig

ENTRYPOINT ["/app/ferrules-api"]
