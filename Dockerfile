FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Build dependencies using recipe.json
FROM chef AS builder
RUN apt-get update -y && apt-get install -y clang
COPY --from=planner /app/recipe.json recipe.json

# Build dependencies - this layer is cached if dependencies don't change
RUN cargo +nightly chef cook --release --recipe-path recipe.json

# Build application
COPY . .
RUN cargo build --release -p ferrules-api

# Runtime stage
FROM debian:bookworm-slim AS runtime
WORKDIR /app

# Install runtime dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary and libs from builder
COPY --from=builder /app/target/release/libonnxruntime*.so /usr/local/lib/
RUN ldconfig
COPY --from=builder /app/target/release/ferrules-api /app/ferrules-api

# Set the entrypoint
ENTRYPOINT ["/app/ferrules-api"]
