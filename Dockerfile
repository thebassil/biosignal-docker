FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod huggingface_hub transformers accelerate safetensors \
    torchaudio scipy scikit-learn numpy yt-dlp librosa nilearn uv

RUN cd /opt && git clone --depth 1 https://github.com/facebookresearch/tribev2.git && \
    cd tribev2 && pip install --no-cache-dir -e .

RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/tribev2', local_dir='/opt/tribev2_weights')"

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
RUN python3 -c "from huggingface_hub import snapshot_download; import os; snapshot_download('meta-llama/Llama-3.2-3B', token=os.environ.get('HF_TOKEN',''))"

COPY handler.py /opt/handler.py

CMD ["python3", "-u", "/opt/handler.py"]
