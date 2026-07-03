FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod huggingface_hub transformers accelerate safetensors \
    torchaudio scipy scikit-learn numpy yt-dlp librosa nilearn

RUN cd /opt && git clone --depth 1 https://github.com/facebookresearch/tribev2.git && \
    cd tribev2 && pip install --no-cache-dir -e .

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/tribev2', local_dir='/opt/tribev2_weights')"

COPY handler.py /opt/handler.py

CMD ["python3", "-u", "/opt/handler.py"]
