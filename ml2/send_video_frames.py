import subprocess
import threading

video_frames_storage = {}
def send_frames_to_mediasoup(frame_generator, target_ip, target_port, width=640, height=480, fps=15):
    print(f"{target_port} - Starting to send frames to Mediasoup at {target_ip}:{target_port}...")
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libvpx",
        "-b:v", "1M",
        "-deadline", "realtime",  # Optimize for low latency
        "-cpu-used", "5",
        "-f", "rtp",
        f"rtp://{target_ip}:{target_port}"
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    try:
        for frame in frame_generator:
            proc.stdin.write(frame)
            proc.stdin.flush()
    except BrokenPipeError:
        print("⚠️ FFmpeg pipe closed, stopping sending.")
    finally:
        proc.stdin.close()
        proc.wait()



def frame_generator(session_id):
    while True:
        frames = video_frames_storage.get(session_id, [])
        if frames:
            frame = frames.pop(0)
            # print(f"📤 Yielding frame for session {session_id}")
            yield frame
        # else:
        #     # print(f"⏳ No frames for session {session_id}")
        #     # time.sleep(0.01)



def stream_video(video_path, target_ip, target_port, width=640, height=480, fps=15):
    print(f"Starting to stream video {video_path} to Mediasoup at {target_ip}:{target_port}...")
    cmd = [
        "ffmpeg",
        "-re",  # Read input at native frame rate
        "-i", video_path,  # Input video file
        "-c:v", "libvpx",  # Use VP8 to match Mediasoup router
        "-b:v", "1M",
        "-s", f"{width}x{height}",  # Scale to specified resolution
        "-r", str(fps),  # Set frame rate
        "-deadline", "realtime",
        "-cpu-used", "5",
        "-f", "rtp",
        f"rtp://{target_ip}:{25001}"
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    try:
        proc.wait()
    except Exception as e:
        print(f"⚠️ Error streaming video: {e}")
    finally:
        proc.terminate()
        print("✅ Video streaming stopped.")