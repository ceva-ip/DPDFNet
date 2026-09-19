FROM python:3.11-slim-bookworm
WORKDIR /bench
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENTRYPOINT ["python", "benchmark.py"]
