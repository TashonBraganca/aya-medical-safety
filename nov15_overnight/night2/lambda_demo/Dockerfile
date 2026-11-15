# Live demo for the judges: one Tiny Aya Global model (Q4_K_M GGUF) behind llama-server, on AWS Lambda (arm64).
# The same image serves three functions; env MODEL picks original | broken | targeted20.
# Lambda Web Adapter forwards Function URL requests to llama-server and streams the answer back.
# llama.cpp is pinned to commit 90e3841, the version that built and judged these files.

FROM --platform=linux/arm64 debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp.git /src && cd /src && git checkout 90e3841
# Graviton2 (Lambda arm64) supports armv8.2-a with dot product; no native tuning so the binary runs on any Lambda arm64 host.
RUN cd /src && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod \
      -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build -j 6 --target llama-server

FROM --platform=linux/arm64 debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /src/build/bin/llama-server /opt/llama-server
COPY chat_template.jinja /opt/chat_template.jinja
COPY start.sh /opt/start.sh
COPY models/ /opt/models/
ENV AWS_LWA_PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/health AWS_LWA_INVOKE_MODE=response_stream AWS_LWA_ASYNC_INIT=true
CMD ["/bin/sh", "/opt/start.sh"]
