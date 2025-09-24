import socketio
import asyncio
import os
import tempfile
import subprocess
from subprocess import Popen
import threading
import time
from pydub import AudioSegment
from pydub.playback import play
import time
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import torch
import torchaudio
from send_video_frames import send_frames_to_mediasoup, frame_generator, video_frames_storage
from fastapi import FastAPI
from pydantic import BaseModel
import wave
import uvicorn
import uuid
import sys
import json
import pickle

# Import whisper streaming functionality
import whisper_utils as wms


#####################################################
import av
import numpy as np

# import cv2
from io import BytesIO


##########################################


ENABLE_TRANSLATION = True
IS_PROD = False

seamlessm4t = 0
if ENABLE_TRANSLATION:
    seamless_streaming = 1
else:
    seamless_streaming = 0

if IS_PROD:
    MEDIASERVER_IP = "10.20.20.35"
else:
    MEDIASERVER_IP = "0.0.0.0"


frames_arrived = False
# Initialize Whisper ASR components globally

whisper_sessions = {}  # session_id -> (whisper_asr, whisper_online)

def initialize_whisper_for_session(session_id, lang="te"):
    """Initialize Whisper ASR for a given session."""
    class WhisperArgs:
        def __init__(self):
            self.language = lang
            self.min_chunk_size = 0.7
            self.vac = True
            self.model = "medium"
            self.task = "translate"
            self.model_cache_dir = None
            self.model_dir = None
            self.backend = "faster-whisper"
            self.vac_chunk_size = 0.05
            self.vad = False
            self.buffer_trimming = "sentance"
            self.buffer_trimming_sec = 15
            self.log_level = "WARNING"

    args = WhisperArgs()
    wms.set_logging(args, wms.logger)

    print(f"Initializing Whisper ASR for session {session_id}...")
    whisper_asr, whisper_online = wms.asr_factory(args, logfile=sys.stderr)

    # Warm-up
    dummy_audio = np.zeros(int(16000), dtype=np.float32)
    whisper_asr.transcribe(dummy_audio)

    whisper_sessions[session_id] = {
        "asr": whisper_asr,
        "online": whisper_online
    }
    print(f"✅ Whisper ASR ready for session {session_id}")


# from streaming_translator_utils import SAMPLE_RATE, StatelessBytesTranslator
# translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output
# start the voice clone
from multiprocessing.connection import Client
import re
def sanitize_speaker_id(speaker_id):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", speaker_id)

def request_voice_clone(audio_tensor, speaker_id="default_male", sample_rate=16000):

    # Ensure it's CPU and numpy
    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.detach().cpu().numpy()
    safe_speaker_id = sanitize_speaker_id(speaker_id)

    conn = Client(("0.0.0.0", 6008), authkey=b"secret_vc")
    conn.send({
        "mode": "clone",
        "audio": audio_tensor,
        "sample_rate": sample_rate,
        "speaker_id": safe_speaker_id,
    })
    result = conn.recv()
    conn.close()
    return result

def request_voice_embed(audio_path, speaker_id="default_male"):
    """
    Sends a voice embedding request using an audio file path.

    Returns True on success, None on failure.
    """
    safe_speaker_id = sanitize_speaker_id(speaker_id)
    conn = Client(('0.0.0.0', 6008), authkey=b'secret_vc')
    conn.send({
        "mode": "embed",
        "audio_path": audio_path,
        "speaker_id": safe_speaker_id
    })
    result = conn.recv()
    conn.close()
    return result





# import pickle
# import json
# def request_lipsync_in_worker(frame_buffer, audio_bytes, output_path):
#     address = ('localhost', 6006)
#     authkey = b'secret'

#     with Client(address, authkey=authkey) as conn:
#         data = pickle.dumps((frame_buffer, audio_bytes, output_path))
#         conn.send_bytes(data)
#         no_audio_path, final_path, error = conn.recv()
#         if error:
#             raise RuntimeError(f"Lip sync failed: {error}")
#         return no_audio_path, final_path


def request_translation_tensor(
    host: str,
    port: int,
    text: str,
    tgt_lang: str,
    timeout: float = 10.0
):
    """
    Send text to Seamless server and receive translated speech as a PyTorch tensor.

    :param host: Server hostname or IP
    :param port: Server port
    :param text: Source text to translate
    :param tgt_lang: Target language code (e.g., 'hin' for Hindi)
    :param timeout: Socket timeout in seconds
    :return: (audio_tensor, sample_rate) or (None, None) on failure
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))

            # Prepare request
            req = json.dumps({"text": text, "tgt_lang": tgt_lang})
            s.sendall(req.encode())

            # Receive length prefix
            length_bytes = s.recv(4)
            if not length_bytes:
                print("[Client] No length prefix received")
                return None, None

            length = int.from_bytes(length_bytes, "big")

            # Receive payload
            data = b""
            while len(data) < length:
                packet = s.recv(length - len(data))
                if not packet:
                    print("[Client] Connection closed before all data received")
                    return None, None
                data += packet

            # Deserialize
            payload = pickle.loads(data)

            audio_tensor = payload["audio_tensor"]
            sample_rate = payload["sample_rate"]
            text_output = payload["text_output"]

            #print(text_output)

            if not isinstance(audio_tensor, torch.Tensor):
                audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

            #print(f"[Client] Received tensor shape: {audio_tensor.shape}, sample rate: {sample_rate}")
            return audio_tensor, sample_rate, text_output

    except Exception as e:
        print(f"[Client] Error: {e}")
        return None, None, None



import webrtcvad
import noisereduce as nr



def bandpass_filter(waveform: np.ndarray, sample_rate: int, lowcut: float = 300.0, highcut: float = 3400.0, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to isolate voice frequencies (300 Hz to 3400 Hz).

    Args:
        waveform (np.ndarray): The mono audio waveform as a 1D float32 NumPy array.
        sample_rate (int): The sampling rate of the audio in Hz.
        lowcut (float): The lower cutoff frequency in Hz (default: 300 Hz).
        highcut (float): The upper cutoff frequency in Hz (default: 3400 Hz).
        order (int): The order of the Butterworth filter (default: 5).

    Returns:
        np.ndarray: The filtered audio waveform as a 1D float32 NumPy array.
    """
    # Calculate the Nyquist frequency (half the sampling rate)
    nyquist = 0.5 * sample_rate

    # Normalize cutoff frequencies to the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the waveform
    filtered_waveform = lfilter(b, a, waveform)

    return filtered_waveform


from collections import defaultdict

#!/usr/bin/env python3
import socket
import struct
import argparse
import time
import numpy as np
import logging
import sys
import re

def convert_audio_format(audio_chunk: bytes, input_sr: int = 48000, target_sr: int = 16000):
    """
    Convert stereo PCM audio from 48kHz to 16kHz mono format for Whisper.
    
    Args:
        audio_chunk: Raw PCM int16 stereo audio bytes at input_sr
        input_sr: Input sample rate (default: 48000)
        target_sr: Target sample rate for Whisper (default: 16000)
    
    Returns:
        numpy.ndarray: Mono float32 audio array at target sample rate
    """
    try:
        # Convert bytes to numpy array (assuming int16 stereo)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Reshape to stereo (2 channels) and convert to mono by averaging
        if len(audio_data) % 2 == 0:
            stereo_data = audio_data.reshape(-1, 2)
            mono_data = np.mean(stereo_data, axis=1)
        else:
            # If odd number of samples, take every other sample (simple downmix)
            mono_data = audio_data[::2]
        
        # Convert to float32 and normalize to [-1, 1]
        mono_float = mono_data.astype(np.float32) / 32768.0
        
        # Resample if needed
        if input_sr != target_sr:
            # Simple resampling using scipy.signal.resample
            target_length = int(len(mono_float) * target_sr / input_sr)
            mono_float = signal.resample(mono_float, target_length)
        
        return mono_float.astype(np.float32)
        
        
    except Exception as e:
        print(f"❗ Error in audio format conversion: {e}")
        # Return silence if conversion fails
        duration_seconds = len(audio_chunk) / (input_sr * 2 * 2)  # stereo int16
        target_samples = int(duration_seconds * target_sr)
        return np.zeros(target_samples, dtype=np.float32)

def get_raw_audio_pcm(samples, sr: int, target_sr: int = 48000, stereo: bool = True):
    import torchaudio

    # Convert input to float32 tensor
    if isinstance(samples, list):
        samples = torch.tensor(samples, dtype=torch.float32)
    elif isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()

    # Mono to [1, N]
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)

    # Convert to stereo if needed
    if stereo and samples.size(0) == 1:
        samples = samples.repeat(2, 1)

    # Resample if needed
    if sr != target_sr:
        samples = torchaudio.functional.resample(samples, orig_freq=sr, new_freq=target_sr)

    # Clamp to [-1, 1]
    samples = torch.clamp(samples, -1.0, 1.0)

    return samples, target_sr  # shape: [2, N]

def is_valid_english(text: str, min_len: int = 2) -> bool:
    """
    Returns True if text contains English letters or numbers
    (not just punctuation) and its length is above min_len.
    Accepts normal sentence punctuation like . , ! ?
    """
    if not text:
        return False

    text_stripped = text.strip()

    # Minimum length check
    if len(text_stripped) < min_len:
        return False

    # Reject if it's only punctuation (e.g., ".", "...", "!!")
    if re.fullmatch(r'[^\w]+', text_stripped):
        return False

    # Accept English letters, numbers, spaces, and common punctuation
    return bool(re.match(r'^[A-Za-z0-9\s.,!?\'"-]+$', text_stripped))

import re
from collections import Counter

def remove_text_loops(text, 
                     max_word_repetition=3, 
                     max_phrase_repetition=2, 
                     max_char_repetition=4,
                     aggressive_mode=False):
    """
    Remove looping patterns from text generated by speech recognition.
    
    Args:
        text (str): Input text to clean
        max_word_repetition (int): Max consecutive word repetitions allowed
        max_phrase_repetition (int): Max consecutive phrase repetitions allowed  
        max_char_repetition (int): Max consecutive character repetitions allowed
        aggressive_mode (bool): Use more aggressive cleaning (slower but thorough)
        
    Returns:
        dict: {
            'original': original text,
            'cleaned': cleaned text, 
            'had_anomaly': bool indicating if loops were detected,
            'anomaly_info': description of detected patterns,
            'reduction_ratio': fraction of text removed,
            'chars_removed': number of characters removed
        }
    """
    
    if not text or not text.strip():
        return {
            'original': text,
            'cleaned': text,
            'had_anomaly': False,
            'anomaly_info': 'Empty text',
            'reduction_ratio': 0,
            'chars_removed': 0
        }
    
    original_text = text.strip()
    cleaned_text = original_text
    
    # Step 1: Remove character loops (fastest)
    char_pattern = r'(.)\1{' + str(max_char_repetition-1) + ',}'
    cleaned_text = re.sub(char_pattern, r'\1' * max_char_repetition, cleaned_text)
    
    # Step 2: Remove word loops
    words = cleaned_text.split()
    if len(words) > 1:
        result = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            count = 1
            
            # Count consecutive repetitions
            while (i + count < len(words) and 
                   words[i + count].lower().strip() == current_word.lower().strip()):
                count += 1
            
            # Add only up to max_repetition instances
            for j in range(min(count, max_word_repetition)):
                result.append(words[i + j])
            
            i += count
        
        cleaned_text = ' '.join(result)
    
    # Step 3: Quick regex-based phrase cleaning
    phrase_patterns = [
        # "phrase and phrase and phrase" -> "phrase and phrase"
        (r'\b(.+?)\s+and\s+\1\s+and\s+\1(?:\s+and\s+\1)*\b', r'\1 and \1'),
        # "phrase phrase phrase" (2-5 words)
        (r'\b((?:\w+\s+){1,4}\w+)(?:\s+\1){2,}\b', r'\1 \1'),
    ]
    
    for pattern, replacement in phrase_patterns:
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
    
    # Step 4: Advanced phrase detection (only if aggressive_mode or text still looks problematic)
    if aggressive_mode or _has_remaining_loops(cleaned_text):
        cleaned_text = _advanced_phrase_cleanup(cleaned_text, max_phrase_repetition)
    
    # Step 5: Clean up extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Detect anomalies
    had_anomaly, anomaly_info = _detect_anomalies(original_text)
    
    # Calculate metrics
    reduction_ratio = 1 - (len(cleaned_text) / len(original_text)) if original_text else 0
    chars_removed = len(original_text) - len(cleaned_text)
    
    return {
        'original': original_text,
        'cleaned': cleaned_text,
        'had_anomaly': had_anomaly,
        'anomaly_info': anomaly_info,
        'reduction_ratio': reduction_ratio,
        'chars_removed': chars_removed
    }

def _has_remaining_loops(text):
    """Quick check for remaining loop patterns"""
    words = text.split()
    if len(words) < 6:
        return False
    
    # Check word diversity
    unique_words = len(set(word.lower() for word in words))
    diversity_ratio = unique_words / len(words)
    
    return diversity_ratio < 0.6  # Low diversity indicates loops

def _advanced_phrase_cleanup(text, max_phrase_repetition):
    """More thorough phrase loop detection (slower)"""
    words = text.split()
    if len(words) < 4:
        return text
    
    result = []
    i = 0
    
    while i < len(words):
        best_match = None
        
        # Try phrase lengths 2-6 words
        for phrase_len in range(2, min(7, len(words) - i + 1)):
            if i + phrase_len > len(words):
                break
            
            phrase = words[i:i + phrase_len]
            phrase_lower = [w.lower().strip() for w in phrase]
            
            # Count consecutive repetitions
            repetitions = 1
            j = i + phrase_len
            
            while j + phrase_len <= len(words):
                next_phrase = words[j:j + phrase_len]
                next_phrase_lower = [w.lower().strip() for w in next_phrase]
                
                if phrase_lower == next_phrase_lower:
                    repetitions += 1
                    j += phrase_len
                else:
                    break
            
            # If we found repetitions, prefer longer phrases
            if repetitions >= 3:
                best_match = {
                    'phrase_len': phrase_len,
                    'repetitions': repetitions,
                    'phrase': phrase
                }
                break  # Take first match (longest phrase due to range order)
        
        # Handle the match
        if best_match:
            # Add only allowed repetitions
            for rep in range(min(best_match['repetitions'], max_phrase_repetition)):
                start_idx = i + (rep * best_match['phrase_len'])
                result.extend(words[start_idx:start_idx + best_match['phrase_len']])
            
            i += best_match['repetitions'] * best_match['phrase_len']
        else:
            result.append(words[i])
            i += 1
    
    return ' '.join(result)

def _detect_anomalies(text):
    """Detect if text has loop patterns"""
    words = text.split()
    if len(words) < 4:
        return False, "Text too short to analyze"
    
    # Word diversity check
    unique_words = len(set(word.lower().strip() for word in words))
    diversity_ratio = unique_words / len(words)
    
    if diversity_ratio < 0.5:
        return True, f"Low word diversity: {diversity_ratio:.2f} ({unique_words}/{len(words)} unique)"
    
    # Dominant word check
    word_counts = Counter(word.lower().strip() for word in words)
    if word_counts:
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        if most_common_count > len(words) * 0.25:
            return True, f"Word '{most_common_word}' dominates: {most_common_count}/{len(words)} ({most_common_count/len(words)*100:.1f}%)"
    
    return False, "No suspicious patterns detected"

# Fast version for real-time processing
def quick_remove_loops(text, max_repetitions=2):
    """
    Ultra-fast loop removal for real-time processing.
    Only does basic word repetition removal.
    
    Estimated latency: <1ms for typical sentences
    """
    if not text or len(text) < 10:
        return text
    
    words = text.split()
    if len(words) <= 2:
        return text
    
    result = []
    i = 0
    
    while i < len(words):
        current_word = words[i]
        count = 1
        
        # Count consecutive identical words (case insensitive)
        while (i + count < len(words) and 
               words[i + count].lower() == current_word.lower()):
            count += 1
        
        # Add up to max_repetitions
        for j in range(min(count, max_repetitions)):
            result.append(words[i + j])
        
        i += count
    
    return ' '.join(result)

# Batch processing for multiple texts
def batch_remove_loops(texts, **kwargs):
    """
    Process multiple texts efficiently
    
    Args:
        texts (list): List of text strings
        **kwargs: Arguments passed to remove_text_loops
        
    Returns:
        list: List of results from remove_text_loops
    """
    return [remove_text_loops(text, **kwargs) for text in texts]


def process_translation_chunk_whisper(
    audio_chunk: bytes,
    output_queue=None,
    input_sr: int = 48000,
    target_sr: int = 16000,
    target_lang: str = None,
    voice_clone_enabled: bool = False,
    session_id=None,
    speaker_id:str = None,
    src_lang: str = None ,
    gender: str = None

):
    """
    Process an already-chunked 1s stereo audio segment through Whisper streaming ASR.

    :param audio_chunk: Raw PCM int16 stereo audio bytes at input_sr (default 48kHz).
    :param output_queue: Optional queue to push transcribed text into.
    :param input_sr: Input sample rate of the audio chunk
    :param target_sr: Target sample rate for Whisper processing
    """
    session_whisper = whisper_sessions.get(session_id)
    
    try:
        start_time = time.time()
        
        # Convert audio format for Whisper processing
        audio_float = convert_audio_format(audio_chunk, input_sr, target_sr)
        
        # Insert audio chunk into Whisper online processor
        if session_whisper and session_whisper["online"]:
            session_whisper["online"].insert_audio_chunk(audio_float)
            #print("seg reached here")

            
            # Process the audio chunk
            result = session_whisper["online"].process_iter()
            #print(result)
            # Extract transcribed text if available
            transcribed_text = wms.output_transcript(result)
            
            if transcribed_text and is_valid_english(transcribed_text):

                #text_clean_time = time.time()
                transcribed_text = remove_text_loops(transcribed_text)['cleaned']
                #print(f"Cleaning time: {time.time()-text_clean_time}")
                print(f"🎤 Transcribed: {transcribed_text}", flush=True)

                translation_time = time.time()
                audio_tensor, sr, text_output = request_translation_tensor(
                    host="0.0.0.0",
                    port=50007,
                    text=transcribed_text,
                    tgt_lang=target_lang
                )
                print(f"seamless time = {time.time() - translation_time}")

                if audio_tensor is not None:
                    print(f"session id :{session_id}, text_output: {text_output}")
                    if voice_clone_enabled:
                        # 🔍 Check if embedding exists
                        speaker_path = os.path.join(
                            "/home/test/docker_image/Docker_voice_clone/voice_embeddings",
                            f"{re.sub(r'[^a-zA-Z0-9_-]', '_', speaker_id)}.pth"
                        )
                        if not os.path.exists(speaker_path):
                            print(f"⚠️ Speaker embedding not found for '{speaker_id}'. Using default.")
                            if gender == "male":
                                speaker_id = "default_male"
                            else: 
                                speaker_id = "default_female"
                        else:
                            speaker_id = speaker_id

                        print(f"Cloning voice for {speaker_id}")
                        clone_tensor = request_voice_clone(audio_tensor, speaker_id=speaker_id, sample_rate=sr)
                        cloned_audio_bytes = tensor_to_bytes(clone_tensor)
                        translated_audio_bytes = resample_audio(cloned_audio_bytes, 22050, 48000)
                    else:
                        samples, sr = get_raw_audio_pcm(audio_tensor, sr)
                        translated_audio_bytes = (samples.numpy() * 32767.0).astype(np.int16).T.tobytes()
                    save_to_wav(translated_audio_bytes, session_id= session_id)
                    output_queue.enqueue(translated_audio_bytes)

            #else:
                #print(f"⚠️ Skipping invalid or non-English transcription: {transcribed_text}", flush=True)

            #else:
                # If no transcription, pass through original audio
            #    output_queue.enqueue(audio_chunk)
        else:
            print("❗ Whisper online processor not initialized")
            output_queue.enqueue(audio_chunk)

        processing_time = time.time() - start_time
        #print(f"⏱️ Whisper processing time: {processing_time:.4f}s")

    except Exception as e:
        print(f"❗ Exception in Whisper processing: {e}")
        # Pass through original audio on error
        if output_queue:
            output_queue.enqueue(audio_chunk)


# %%

SAMPLE_READ_SIZE = 4096  # minimum number of bytes read from the audio buffers/arrays
OUTPUT_PERIOD = 0.02  # defines frequency at which output is written to the network


# ----------------- OutputAudioQueue ----------------- #
# class responsible for handling the queue used to output audio with thread safety
class OutputAudioQueue:
    def __init__(self):
        self.data = bytearray()  # Array that stores the audio queue
        self.lock = threading.Lock()  # Lock used to control access between threads
        self.closed = False  # Indicates when the process must be stopped
        self.timeout_seconds = 10 * 60  # Stops the threads after 10 min of inactivity
        self.last_write = time.perf_counter()  # Saves last enqueue time

    # Appends new data to the queue
    def enqueue(self, new_data: bytes):
        with self.lock:
            self.data.extend(new_data)
            self.last_write = time.perf_counter()

    # Reads and removes the specified number of bytes from the queue
    def dequeue(self, size):
        with self.lock:
            if time.perf_counter() - self.last_write > self.timeout_seconds:
                self.closed = True
            if len(self.data) == 0:
                # print("returning empty bytes to client")
                return b""
            if size > len(self.data):
                size = len(self.data)
            dequeued_data = self.data[:size]
            self.data = self.data[size:]
            return dequeued_data


# ----------------- Utilities ----------------- #


# Saves the bytes to a wav file on disk for debugging
def save_to_wav(audio_bytes: bytes, sample_rate=48000, num_channels=2, sample_width=2, session_id:str= None):
    BASE_DIR = "/home/cb-translator/ml/recordings"  # change to your real full path
    session_path = os.path.join(BASE_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    filename = os.path.join(session_path, f"{int(time.time() * 1000)}.wav")
    print(f"saving output at {filename}")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    print(f"💾 Saved audio segment to {filename}")


# Initializes the file used to read input from the network
def write_sdp_file(payload_type, codec_name, clock_rate, channels, rtp_port):
    sdp_content = f"""v=0
o=- 0 0 IN IP4 0.0.0.0
s=Mediasoup Audio
c=IN IP4 0.0.0.0
t=0 0
m=audio {rtp_port} RTP/AVP {payload_type}
a=rtpmap:{payload_type} {codec_name}/{clock_rate}/{channels}
a=recvonly
""".strip()

    tmp_dir = tempfile.gettempdir()
    sdp_path = os.path.join(tmp_dir, f"audio_{int(os.getpid())}.sdp")

    with open(sdp_path, "w") as f:
        f.write(sdp_content)

    return sdp_path

def run_ffmpeg(sdp_path):
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel", "info",
            "-protocol_whitelist", "file,udp,rtp",
            "-f", "sdp",
            "-i", sdp_path,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            "-f", "wav",
            "pipe:1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# Creates pipe that reads the data from the provided SDP path
def run_ffmpeg_input(sdp_path):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "info",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-f",
            "sdp",
            "-i",
            sdp_path,
            "-c:a",
            "pcm_s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-f",
            "s16le",  # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )



# Creates pipe that writes to the destination RTP endpoint
def run_ffmpeg_output(target_ip: str, target_port: int, payload_type: int, ssrc: int):
    cmd = [
        "ffmpeg",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-payload_type",
        str(payload_type),
        "-ssrc",
        str(ssrc),
        "-f",
        "rtp",
        f"rtp://{target_ip}:{target_port}",
    ]
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


# Logs FFmpeg errors
def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b""):
        text = line.decode(errors="ignore").strip()
        if "error" in text.lower():
            print(f"{label}: {text}")


# Resamples and converts mono to stereo
def resample_audio(audio_bytes, original_sr=16000, target_sr=48000):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    new_length = int(len(audio_data) * target_sr / original_sr)
    resampled = signal.resample(audio_data, new_length)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    stereo_data = np.column_stack((resampled, resampled)).flatten()
    return stereo_data.tobytes()


# Converts a numpy tensor to raw PCM bytes
def tensor_to_bytes(translated_wav):
    audio_np = np.clip(np.array(translated_wav, dtype=np.float32), -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()



import numpy as np
import librosa

def is_silent(audio_tensor, threshold_db=-45.0, zcr_threshold=0.12):
    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.detach().cpu().numpy()

    # 1. RMS in dB
    rms = np.sqrt(np.mean(np.square(audio_tensor)))
    db = 20 * np.log10(rms + 1e-6)
    #print(f"🔊 RMS dB: {db:.2f}")

    # 2. Quick Zero-Crossing Rate
    zero_crossings = np.sum(np.diff(np.sign(audio_tensor)) != 0)
    zcr = zero_crossings / len(audio_tensor)
    #print(f"🔄 ZCR: {zcr:.4f}")

    # Determine silence or noise
    return db < threshold_db and zcr > zcr_threshold

def process_voice_embedding_chunk(
    seg,
    session_id,
    chunk_count,
    audio_buffer,
    tensor_buffer,
    embedding_counter,
    speaker_id,  # 🔸 current speaker_id passed in
    save_audio,
    request_voice_embed,
    bytes_to_float32_mono_array,
    bandpass_filter,
    nr,
    is_silent
):
    audio_buffer.append(seg)
    print(session_id, chunk_count)
    save_time = time.time()

    if chunk_count == 50:
        os.makedirs("/home/cb-translator/ml/recordings", exist_ok=True)
        print("🟡 Received 50 chunks – checking for silence")

        for chunk in audio_buffer:
            try:
                mono_tensor = bytes_to_float32_mono_array(chunk)
                mono_tensor = bandpass_filter(mono_tensor, 16000)
                mono_tensor = nr.reduce_noise(y=mono_tensor, sr=16000)
                if not is_silent(mono_tensor):
                    tensor_buffer.append(mono_tensor)
            except Exception as e:
                print(f"⚠️ Failed to process chunk: {e}")

        if len(tensor_buffer) == 0:
            print("❌ All chunks were silent. Skipping embedding.")
            audio_buffer.clear()
            tensor_buffer.clear()
            return 0, embedding_counter, speaker_id  # Return unchanged speaker_id

        # ✅ Do embedding
        # ✅ Convert in place
        for i in range(len(tensor_buffer)):
            if isinstance(tensor_buffer[i], np.ndarray):
                tensor_buffer[i] = torch.tensor(tensor_buffer[i])        
        #print(f"audio buffer length{len(tensor_buffer)}")
        stitched_tensor = torch.cat(tensor_buffer, dim=-1).to(torch.float32)

        saved_path = save_audio(stitched_tensor, sample_rate=16000, prefix=f"{session_id}")
        print(saved_path)
        #new_speaker_id = f"{session_id}"
        emb_time = time.time()
        request_voice_embed(saved_path, speaker_id)
        print(f"Time taken to request voice embedding: {time.time() - emb_time}")
        print(f"✅ Voice embedding done for {speaker_id}")
        embedding_counter += 1

        #try:
        #    os.remove(saved_path)
        #    print(f"🧹 Deleted temp audio file: {saved_path}")
        #except Exception as e:
        #    print(f"⚠️ Failed to delete temp audio file: {e}")

        audio_buffer.clear()
        tensor_buffer.clear()
        #print(f"audio buffer length after clearning {len(tensor_buffer)}")

        return 0, embedding_counter, speaker_id  # ✅ Return updated speaker_id
    # Not enough chunks yet
    return chunk_count, embedding_counter, speaker_id

def bytes_to_float32_mono_array(audio_bytes: bytes, input_sr=48000, target_sr=16000) -> np.ndarray:
    # 1. Decode stereo int16 bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Debug: print raw input size and dtype
    #print(f"📦 Raw audio buffer: {audio_np.shape}, dtype: {audio_np.dtype}")

    if len(audio_np.shape) == 1 and len(audio_np) % 2 == 0:
        audio_np = audio_np.reshape(-1, 2)  # 2 channels
    elif audio_np.shape[-1] != 2:
        print("❗ Warning: Audio shape not 2-channel. Padding mono to stereo.")
        audio_np = np.stack([audio_np, audio_np], axis=-1)

    # 2. Convert to mono by averaging channels
    mono_np = audio_np.mean(axis=1)

    # Print min/max and energy before normalization
    #print(f"🔊 Min: {mono_np.min()}, Max: {mono_np.max()}, Mean: {mono_np.mean():.4f}")

    # 3. Convert to float32 [-1.0, 1.0]
    waveform = torch.tensor(mono_np, dtype=torch.float32) / 32768.0
    waveform = waveform.unsqueeze(0)  # (1, N)

    # Print energy and duration
    duration = waveform.shape[1] / input_sr
    #print(f"📏 Duration: {duration:.2f}s, Energy (RMS): {torch.sqrt(torch.mean(waveform**2)):.6f}")

    # 4. Resample
    resampled = torchaudio.functional.resample(waveform, orig_freq=input_sr, new_freq=target_sr)
    return resampled.squeeze(0).numpy()

def save_audio(audio_tensor, sample_rate=16000, prefix="embedding"):
    # Remove anything after '@'
    if "@" in prefix:
        prefix = prefix.split("@")[0]

    filename = f"{prefix}_{time.time()}.wav"
    os.makedirs("recordings", exist_ok=True)
    relative_path = os.path.join("recordings", filename)
    full_path = os.path.abspath(relative_path)  # 🔁 Convert to absolute path

    # Ensure correct shape and dtype for torchaudio
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.to(torch.float32)

    torchaudio.save(full_path, audio_tensor.cpu(), sample_rate)
    return full_path



# Function that reads from the input pipe, processes audio, and enqueues to output
def pump_audio(
    ff_in: Popen,
    ff_out: Popen,
    output_queue: OutputAudioQueue,
    segment_size: int,
    sample_rate: int,
    sdp_path: str,
    target_lang,
    session_id: str,
    speaker_id: str,
    src_lang,
    gender: str,
    voice_clone_enabled: bool = True,
):
    buf = b""
    audio_buffer = []
    tensor_buffer = []
    chunk_count = 0
    embedding_counter = 0
    empty_chunk_counter = 0   # 🔹 track consecutive empty chunks

    try:
        while True:
            chunk = ff_in.stdout.read(SAMPLE_READ_SIZE)
            if not chunk:
                empty_chunk_counter += 1
                print(f"⚠️ Empty chunk detected for session {session_id} ({empty_chunk_counter}/20)")
                if empty_chunk_counter >= 20:
                    print(f"🛑 No audio for 20 loops, clearing Whisper session {session_id}")
                    whisper_sessions.pop(session_id, None)   # 🔹 cleanup
                    break
                time.sleep(0.05)  # avoid tight loop
                continue
            else:
                empty_chunk_counter = 0  # reset on valid chunk

            if output_queue.closed:
                print("output closed, stopping")
                break

            buf += chunk
            while len(buf) >= segment_size:
                chunk_count += 1
                seg, buf = buf[:segment_size], buf[segment_size:]                
                #print(chunk_count)
                #save_to_wav(seg, session_id=session_id)
                start_time = time.time()
                #voice_clone_enabled = False
                #####################################################################################
                if voice_clone_enabled:
                    # 🔁 Inside your while loop:
                    chunk_count, embedding_counter, speaker_id = process_voice_embedding_chunk(
                        seg=seg,
                        session_id=session_id,
                        chunk_count=chunk_count,
                        audio_buffer=audio_buffer,
                        tensor_buffer=tensor_buffer,
                        embedding_counter=embedding_counter,
                        speaker_id=speaker_id,  # ✅ Pass current speaker_id
                        save_audio=save_audio,
                        request_voice_embed=request_voice_embed,
                        bytes_to_float32_mono_array=bytes_to_float32_mono_array,
                        bandpass_filter=bandpass_filter,
                        nr=nr,
                        is_silent=is_silent,
                    )
                ##############################################################

                if seamless_streaming == 1:
                    process_translation_chunk_whisper(
                        seg,                  # audio chunk bytes
                        output_queue,         # your queue to receive transcriptions
                        input_sr=48000,       # input sample rate of the audio chunk
                        target_sr=16000,       # sample rate expected by Whisper
                        target_lang = target_lang,
                        voice_clone_enabled = voice_clone_enabled,
                        session_id=session_id,
                        speaker_id=speaker_id,
                        src_lang = src_lang,
                        gender = gender,
                    )
                    #################################################
                else:
                    video_frames = video_frames_storage.pop(session_id, None)
                    print(
                        f"🟩 Processing video and audio segment for session {session_id}... {chunk_count}"
                        + (
                            f", video frames: {len(video_frames)}"
                            if video_frames is not None
                            else ", no video frames found."
                        )
                    )
                    
                    
                    output_queue.enqueue(seg)
                #print(f"Processing time: {(time.time() - start_time):.4f} seconds")
    finally:
        output_queue.closed = True
        ff_in.stdout.close()
        ff_out.stdin.close()
        ff_in.wait()
        ff_out.wait()
        try:
            os.remove(sdp_path)
        except OSError:
            pass


# Function that writes translated audio from the output queue to the output pipe at correct throughput
def write_to_output(output_queue: OutputAudioQueue, ff_out: Popen):
    next_time = time.perf_counter()
    try:
        while not output_queue.closed:
            seg = output_queue.dequeue(SAMPLE_READ_SIZE)
            if seg:
                try:
                    # print("=======================", seg)
                    ff_out.stdin.write(seg)
                    ff_out.stdin.flush()
                except BrokenPipeError:
                    # stop the voice clone
                    print("⚠️ FFmpeg-OUT pipe closed")
                    return
            next_time += OUTPUT_PERIOD
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()
    except Exception as e:
        print(f"Error in processing thread: {e}")
    finally:
        output_queue.closed = True


# ----------------- FastAPI Server ----------------- #
app = FastAPI()


class TranslationRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    channels: int
    rtpPort: int
    outputPort: int
    ssrc: int
    targetLang: str
    sessionId: str
    userId: str
    gender: str
    srcLang: str
    enableVoiceClone: bool



is_first = True


@app.post("/translation/initiate")
async def initiate_translation(data: TranslationRequest):
    global system
    print("📥 Received translation initiation:", data.dict())
    sample_rate = data.clockRate
    if ENABLE_TRANSLATION:
        if data.sessionId not in whisper_sessions:
            initialize_whisper_for_session(data.sessionId, lang="auto")  # or based on target_lang
        else:
            print(f"🔁 Reusing existing system for session: {data.sessionId}")

    time.sleep(0.2)

    # Sets up the read file from the rtp port provided by the client
    sdp_path = write_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=sample_rate,
        channels=data.channels,
        rtp_port=data.rtpPort,
    )

    ff_in = run_ffmpeg_input(sdp_path)
    ff_out = run_ffmpeg_output(
        MEDIASERVER_IP, data.outputPort, data.payloadType, data.ssrc
    )

    # Create threads that log errors encountered by FFmpeg
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_in, "FFmpeg-IN"), daemon=True
    ).start()
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_out, "FFmpeg-OUT"), daemon=True
    ).start()

    if ENABLE_TRANSLATION:
        segment_size = int(sample_rate * 2 * 2 * 1)
    else:
        segment_size = int(sample_rate * 2 * 2 * 2)

    # Initializes output audio queue
    output_queue = OutputAudioQueue()
    target_lang = data.targetLang
    #src_lang = "te"#src_lang = data.srcLang
    src_lang = data.srcLang
    #gender = "male"#gender = data.gender
    gender = data.gender
    #print(target_lang)
    #target_lang = "hin"
    # manually enter the language code here
    # Create thread to process audio
    threading.Thread(
        target=pump_audio,
        args=(
            ff_in,
            ff_out,
            output_queue,
            segment_size,
            sample_rate,
            sdp_path,
            target_lang,
            data.sessionId,
            data.userId,#data.sessionId.split("@")[0] #data.userId #user id  
            src_lang,
            gender,
            data.enableVoiceClone,
        ),
        daemon=True,
    ).start()

    # Create thread to output audio
    threading.Thread(
        target=write_to_output, args=(output_queue, ff_out), daemon=True
    ).start()

    return {"status": "Translation pipeline started"}


def write_video_sdp_file(payload_type, codec_name, clock_rate, rtp_port):
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=Mediasoup Video\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        f"m=video {rtp_port} RTP/AVP {payload_type}\n"
        f"a=rtpmap:{payload_type} {codec_name}/{clock_rate}\n"
        "a=recvonly\n"
        "a=rtcp-mux\n"
    )
    fn = f"video_{uuid.uuid4().hex}.sdp"
    path = os.path.join(tempfile.gettempdir(), fn)
    with open(path, "w") as f:
        f.write(sdp)
    return path


def run_ffmpeg_video_pipe(sdp_path, width=640, height=480):
    print(f"Running FFmpeg with SDP path: {sdp_path}")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "info",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-f",
        "sdp",
        "-i",
        sdp_path,
        "-an",  # no audio
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8
    )


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # for bgr24
FPS = 250
NUM_OF_SECONDS = 5



def store_frames(
    proc: Popen,
    frame_width: int,
    frame_height: int,
    session_id: str = None,
    video_frames_storage=video_frames_storage,
):
    frame_size = frame_width * frame_height * 3  # BGR24
    count = 0
    try:
        while True:
            count += 1
            raw_frame = proc.stdout.read(frame_size)
            # print(f"📥 Received {len(raw_frame)} bytes for session {session_id} {count}")
            if not raw_frame:
                print("📤 FFmpeg pipe ended")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (frame_height, frame_width, 3)
            )
            # print(f"📦 Received frame for session {session_id} ({frame.shape})")
            video_frames_storage.setdefault(session_id, []).append(frame)


    except Exception as e:
        print(f"⚠️ Error in store_frames: {e}")
    finally:
        try:
            proc.stdout.close()
            proc.stderr.close()
            proc.terminate()
            proc.wait(timeout=5)
        except:
            pass
        print("✅ Frame storage stopped")

def store_frames_as_bytes(proc: Popen):
    """
    This function is a placeholder for storing frames as bytes.
    It can be implemented to convert frames to bytes and store them in a suitable format.
    """
    pass

class VideoCaptureRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    rtpPort: int
    targetPort: int
    sessionId: str


@app.post("/video/initiate")
async def initiate_video_capture(data: VideoCaptureRequest):
    return
    print("📥 Received video capture initiation:", data.dict())

    sdp_path = write_video_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=data.clockRate,
        rtp_port=data.rtpPort,
    )

    ffmpeg_proc = run_ffmpeg_video_pipe(
        sdp_path, width=FRAME_WIDTH, height=FRAME_HEIGHT
    )
    
    print(f"🔄️ FFmpeg process started with PID {ffmpeg_proc.pid}")

    threading.Thread(
        target=print_ffmpeg_logs, args=(ffmpeg_proc, "FFmpeg-VIDEO"), daemon=True
    ).start()

    
    threading.Thread(
        target=store_frames,
        args=(
            ffmpeg_proc,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            data.sessionId,
            video_frames_storage,
        ),
        daemon=True,
    ).start()
    
    # threading.Thread(
    #     target=send_frames_to_mediasoup,
    #     args=(
    #         frame_generator(data.sessionId),
    #         MEDIASERVER_IP,  # your Mediasoup plain transport IP
    #         data.targetPort  # your Mediasoup plain transport video port
    #     ),
    #     daemon=True,
    # ).start()

    return {"status": "Frame-based video capture started."}


# 640 x 480


if __name__ == "__main__":
    uvicorn.run("_main:app", host="0.0.0.0", port=2004, reload=False)
# %%
