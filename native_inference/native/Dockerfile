FROM dpdfnet-native-probe
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir soundfile==0.13.1
ENTRYPOINT ["python", "native/probe.py"]
