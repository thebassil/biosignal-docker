"""
RunPod Serverless handler for TRIBE v2 inference.
Uses yt-dlp for YouTube downloads (updated version).
Falls back to HTTP GET for direct URLs.
"""
import runpod
import numpy as np
import json
import subprocess
import tempfile
import os
import sys
import time
import traceback

sys.path.insert(0, "/opt/tribev2")

MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    import torch
    from tribev2 import TribeModel
    print("Loading TRIBE v2...", flush=True)
    MODEL = TribeModel.from_pretrained("/opt/tribev2_weights", cache_folder="/tmp/tribe_cache")
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    return MODEL

def download_audio(url_or_query, output_dir):
    """Download audio via yt-dlp or HTTP, convert to 16kHz mono WAV."""
    safe = url_or_query.replace(" ", "_").replace("/", "-").replace("'", "")[:60]
    wav_path = os.path.join(output_dir, f"{safe}.wav")

    # Try HTTP GET first for direct URLs (S3, CDN, etc)
    if url_or_query.startswith("http") and "youtube" not in url_or_query and "ytsearch" not in url_or_query:
        try:
            from urllib.request import urlopen, Request
            req = Request(url_or_query, headers={"User-Agent": "biosignal/1.0"})
            resp = urlopen(req, timeout=120)
            raw = os.path.join(output_dir, "raw_dl")
            with open(raw, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk: break
                    f.write(chunk)
            if os.path.getsize(raw) > 5000:
                subprocess.run(["ffmpeg","-i",raw,"-ar","16000","-ac","1","-t","300",wav_path,"-y","-loglevel","error"], timeout=60)
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 5000:
                    return wav_path
        except Exception as e:
            print(f"HTTP download failed: {e}", flush=True)

    # yt-dlp for YouTube URLs and ytsearch queries
    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1 -t 300",
            "-o", os.path.join(output_dir, f"{safe}.%(ext)s"),
            "--no-playlist", "--max-downloads", "1",
            url_or_query,
        ], capture_output=True, timeout=120)

        for ext in [".webm", ".opus", ".m4a"]:
            src = os.path.join(output_dir, f"{safe}{ext}")
            if os.path.exists(src) and not os.path.exists(wav_path):
                subprocess.run(["ffmpeg", "-i", src, "-ar", "16000", "-ac", "1",
                                wav_path, "-y", "-loglevel", "error"], timeout=60)
                os.unlink(src)

        return wav_path if os.path.exists(wav_path) and os.path.getsize(wav_path) > 5000 else None
    except Exception as e:
        print(f"Download error: {e}", flush=True)
        return None

def extract_destrieux_features(preds):
    """Extract Destrieux parcellation features from raw (T, 20484) predictions."""
    from scipy import stats as sp

    T, V = preds.shape
    features = {}

    # Load Destrieux atlas for fsaverage5
    try:
        from nilearn import datasets
        atlas = datasets.fetch_atlas_surf_destrieux()
        labels_lh = atlas['labels_lh'] if hasattr(atlas['labels_lh'], '__len__') else np.array(atlas['labels_lh'])
        labels_rh = atlas['labels_rh'] if hasattr(atlas['labels_rh'], '__len__') else np.array(atlas['labels_rh'])
        all_labels = np.concatenate([labels_lh, labels_rh])
        label_names = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas['labels']]
        use_atlas = True
    except Exception as e:
        print(f"Atlas load failed ({e}), using Yeo-7 approximation", flush=True)
        use_atlas = False

    if use_atlas and len(all_labels) == V:
        # Real Destrieux parcellation
        unique_labels = sorted(set(all_labels))
        for label_idx in unique_labels:
            if label_idx == 0: continue  # skip medial wall
            if label_idx < len(label_names):
                name = label_names[label_idx].replace("-", "_").replace(".", "_")
            else:
                name = f"region_{label_idx}"
            
            mask = all_labels == label_idx
            if mask.sum() < 5: continue
            
            region_ts = preds[:, mask].mean(axis=1)
            
            # Full temporal stats
            features[f"region_{name}_mean"] = float(region_ts.mean())
            features[f"region_{name}_std"] = float(region_ts.std())
            if T > 2:
                slope, _, _, _, _ = sp.linregress(np.arange(T), region_ts)
                features[f"region_{name}_slope"] = float(slope)
                features[f"region_{name}_range"] = float(region_ts.max() - region_ts.min())
            
            # Temporal windows
            if T >= 10:
                # First/last 30s (assuming ~2s per timepoint)
                win = min(15, T//3)
                for wname, sl in [("first30s", slice(0, win)), ("last30s", slice(-win, None)), ("peak30s", None)]:
                    if wname == "peak30s":
                        peak_idx = np.argmax(np.abs(region_ts))
                        start = max(0, peak_idx - win//2)
                        end = min(T, start + win)
                        sl = slice(start, end)
                    wts = region_ts[sl]
                    features[f"region_{name}_mean_{wname}"] = float(wts.mean())
                    features[f"region_{name}_std_{wname}"] = float(wts.std())
                    if len(wts) > 2:
                        s, _, _, _, _ = sp.linregress(np.arange(len(wts)), wts)
                        features[f"region_{name}_slope_{wname}"] = float(s)
                
                # 5 equal windows
                wsize = T // 5
                for wi in range(5):
                    wts = region_ts[wi*wsize:(wi+1)*wsize]
                    features[f"region_{name}_mean_w{wi+1}of5"] = float(wts.mean())
                    features[f"region_{name}_slope_w{wi+1}of5"] = float(sp.linregress(np.arange(len(wts)), wts)[0]) if len(wts) > 2 else 0.0

    # Yeo-7 network features (always computed)
    n = V
    boundaries = [0, n//7, 2*n//7, 3*n//7, 4*n//7, 5*n//7, 6*n//7, n]
    yeo7 = ["visual", "somatomotor", "dorsal_attention", "ventral_attention", "limbic", "control", "default_mode"]

    network_ts = []
    for i, name in enumerate(yeo7):
        ts = preds[:, boundaries[i]:boundaries[i+1]].mean(axis=1)
        network_ts.append(ts)
        features[f"spatial_{name}"] = float(ts.mean())
        features[f"stability_{name}_std"] = float(ts.std())
        features[f"stability_{name}_range"] = float(ts.max() - ts.min())
        if T > 2:
            slope, _, _, _, _ = sp.linregress(np.arange(T), ts)
            features[f"stability_{name}_slope"] = float(slope)

    network_ts = np.array(network_ts)
    for i in range(7):
        for j in range(i+1, 7):
            r, _ = sp.pearsonr(network_ts[i], network_ts[j])
            features[f"coupling_{yeo7[i]}_{yeo7[j]}"] = float(r)
    corr = np.corrcoef(network_ts)
    features["coupling_mean_corr"] = float(corr[np.triu_indices(7, k=1)].mean())

    features["global_mean"] = float(preds.mean())
    features["global_std"] = float(preds.std())
    global_ts = preds.mean(axis=1)
    if T > 2:
        features["global_autocorr"] = float(np.corrcoef(global_ts[:-1], global_ts[1:])[0, 1])
    if T > 1:
        cos = [float(np.dot(preds[t], preds[t+1]) / (np.linalg.norm(preds[t]) * np.linalg.norm(preds[t+1]) + 1e-8)) for t in range(T-1)]
        features["stability_cos_mean"] = float(np.mean(cos))
        features["stability_cos_std"] = float(np.std(cos))
    if T > 10:
        fft = np.abs(np.fft.rfft(global_ts))
        freqs = np.fft.rfftfreq(T, d=2.0)
        total = fft.sum() + 1e-8
        features["freq_slow_power_ratio"] = float(fft[(freqs >= 0.01) & (freqs < 0.05)].sum() / total)
        features["freq_medium_power_ratio"] = float(fft[(freqs >= 0.05) & (freqs < 0.15)].sum() / total)
        features["freq_fast_power_ratio"] = float(fft[freqs >= 0.15].sum() / total)
    features["pleasure_static"] = features.get("spatial_limbic", 0)
    features["pleasure_std"] = features.get("stability_limbic_std", 0)
    features["pleasure_range"] = features.get("stability_limbic_range", 0)
    features["pleasure_deriv_var"] = float(np.var(np.diff(network_ts[4])))
    features["coupling_aud_reward"] = features.get("coupling_somatomotor_limbic", 0)

    return features

def handler(job):
    try:
        inp = job["input"]
        audio_url = inp.get("audio_path") or inp.get("audio_url", "")
        song_name = inp.get("song_name", "unknown")
        group = inp.get("group", "unknown")
        genre = inp.get("genre", "unknown")

        # Check volume first
        if inp.get("audio_path") and os.path.exists(inp["audio_path"]):
            wav_path = inp["audio_path"]
            audio_source = "volume"
        else:
            model = load_model()  # ensure loaded before download
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = download_audio(audio_url, tmpdir)
                if wav_path is None:
                    return {"status": "error", "error": "download_failed", "song_name": song_name}
                audio_source = "download"

        model = load_model()
        t0 = time.time()
        df = model.get_events_dataframe(audio_path=wav_path)
        preds, segs = model.predict(events=df)
        dt = time.time() - t0

        features = extract_destrieux_features(preds)
        features["group"] = group
        features["genre"] = genre

        # Persist to volume if available
        try:
            results_dir = "/runpod-volume/results"
            if os.path.isdir("/runpod-volume"):
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, f"{job.get('id','unknown')}.json"), "w") as f:
                    json.dump({"status": "success", "song_name": song_name, "group": group,
                               "genre": genre, "features": features, "predictions_shape": list(preds.shape),
                               "inference_time": round(dt, 1)}, f)
        except: pass

        return {"status": "success", "song_name": song_name, "group": group, "genre": genre,
                "predictions_shape": list(preds.shape), "inference_time": round(dt, 1),
                "audio_source": audio_source, "features": features}

    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
