# Linux test environment for platforms the pinned torch CPU wheel is not built for, such as macOS.
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md requirements.txt requirements-dev.txt ./
COPY careaccess ./careaccess
RUN python -m pip install --upgrade "pip>=26.2.1" "setuptools>=83.0.0" \
    && python -m pip install --no-cache-dir -r requirements-dev.txt
COPY . .

CMD ["python", "-m", "pytest", "-q"]
