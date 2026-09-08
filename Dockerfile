FROM python:3.12-slim AS runtime

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["personal-agent-memory"]

FROM runtime AS acceptance

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y \
        chromium chromium-driver fonts-noto-cjk nodejs \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir '.[acceptance]'
COPY plugins ./plugins

ENTRYPOINT ["python"]
