FROM runpod/pytorch:2.4.1-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod huggingface_hub transformers accelerate safetensors \
    torchaudio scipy scikit-learn numpy yt-dlp librosa nilearn

RUN cd /opt && git clone https://github.com/facebookresearch/tribev2.git && \
    cd tribev2 && pip install --no-cache-dir -e ".[plotting]"

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/tribev2', local_dir='/opt/tribev2_weights')"

COPY handler.py /opt/handler.py

CMD ["python3", "-u", "/opt/handler.py"]
