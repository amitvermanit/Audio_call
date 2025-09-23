# voice_clone_worker.py
import torchaudio
from multiprocessing.connection import Listener
from TTS.api import TTS
import time
import torch
from pathlib import Path
import scipy
import numpy as np
import os
from custom_openvoice import OpenVoice

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
print("📦 Loading model...")
tts = TTS(model_name="voice_conversion_models/multilingual/multi-dataset/openvoice_v2").cuda()
vc_model: OpenVoice = tts.voice_converter.vc_model
vc_model.__class__ = OpenVoice
vc_model = vc_model.to("cuda")
print("✅ Model loaded!")

def save_to_wav(wav):
    path = "testOutputs/"
    os.makedirs(path, exist_ok=True)
    path +=f"{time.time()}.wav"

    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(path, vc_model.config.audio.output_sample_rate, wav_norm)

def preprocess_live_audio_for_clone(audio_data, orig_sr, target_sr):
    """Resample and normalize audio tensor for cloning"""
    if isinstance(audio_data, list):
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
    elif isinstance(audio_data, np.ndarray):
        audio_data = torch.from_numpy(audio_data).float()
    if not isinstance(audio_data, torch.Tensor):
        raise TypeError("audio_data must be a list, np.ndarray, or torch.Tensor")

    if audio_data.dim() > 1:
        audio_data = torch.mean(audio_data, dim=0)

    audio_data = audio_data / audio_data.abs().max().clamp(min=1e-5)

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=vc_model.config.audio.input_sample_rate)
    return resampler(audio_data.unsqueeze(0)).squeeze(0)

def voice_clone_server():
    address = ('0.0.0.0', 6008)
    listener = Listener(address, authkey=b'secret_vc')
    print("🟢 Voice clone server is listening on port 6008...")

    while True:
        conn = listener.accept()
        try:
            data = conn.recv()
            mode = data.get("mode", "clone")
            speaker_id = data["speaker_id"]
            print(f"🎧 Mode: {mode} | Speaker ID: {speaker_id}")

            if mode == "clone":
                audio_data = data["audio"]
                sample_rate = data.get("sample_rate", 16000)

                start_time = time.time()
                preprocessed = preprocess_live_audio_for_clone(audio_data, sample_rate, vc_model.config.audio.input_sample_rate)
                converted_wav = vc_model.voice_conversion(
                    preprocessed.to(vc_model.device),
                    speaker_id=speaker_id,
                    voice_dir=os.path.abspath("./voice_embeddings")
                )
                print(f"✅ Clone complete in {time.time() - start_time:.2f}s")
                #print(type(converted_wav))
                #save_to_wav(converted_wav)               
                conn.send(converted_wav)

            elif mode == "embed":
                audio_path = data["audio_path"]
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"❌ File not found: {audio_path}")

                start_time = time.time()
                vc_model.voice_conversion(
                    audio_path,
                    audio_path,  # dummy target
                    speaker_id=speaker_id,
                    voice_dir=os.path.abspath("./voice_embeddings")
                )
                print(f"✅ Embedding saved in {time.time() - start_time:.2f}s")
                conn.send(True)

            else:
                print(f"❌ Unknown mode: {mode}")
                conn.send(None)

        except Exception as e:
            print(f"❌ Error: {e}")
            conn.send(None)
        finally:
            conn.close()

if __name__ == "__main__":
    voice_clone_server()
