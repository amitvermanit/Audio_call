import socket
import json
import torch
import pickle
from seamless_communication.inference import Translator

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
translator = Translator(model_name, vocoder_name, device=device, dtype=torch.float16)
print("translator loaded")

host = '0.0.0.0'
port = 50007

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"[Seamless Server] Listening on {host}:{port}")

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"[Seamless Server] Connected by {addr}")

            try:
                # Receive request JSON
                data = conn.recv(4096).decode()
                req = json.loads(data)
                input_text = req["text"]
                tgt_lang = req["tgt_lang"]
            except Exception as e:
                print(f"[ERROR] Invalid request: {e}")
                conn.sendall(pickle.dumps({"error": "Invalid request"}))
                continue

            # Translation
            text_output, speech_output = translator.predict(
                input=input_text,
                task_str="t2st",
                tgt_lang=tgt_lang,
                src_lang="eng"
            )

            audio_tensor = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
            sample_rate = speech_output.sample_rate

            # Package into dict
            # Package into dict including text output
            payload = {
                "sample_rate": sample_rate,
                "audio_tensor": audio_tensor,
                "text_output": str(text_output)  # <-- add text output
            }

            # Serialize with pickle
            data_bytes = pickle.dumps(payload)

            # Send length + payload
            conn.sendall(len(data_bytes).to_bytes(4, "big"))
            conn.sendall(data_bytes)

            print(f"[Seamless Server] Sent tensor with shape {audio_tensor.shape} @ {sample_rate} Hz")
            print(f"[Seamless Server] Text Output: {text_output}")
