import os
import wave
from datetime import datetime

import pyaudio

# =====================================================
# 1. Recording parameters (matched to your hardware)
# =====================================================

SAMPLE_RATE = 96000        # Hz (Focusrite Scarlett 2i2)
CHANNELS = 1               # Mono
DURATION = 10              # seconds
CHUNK = 1024               # frames per buffer
FORMAT = pyaudio.paFloat32

# =====================================================
# 2. Output path: Desktop/PA/sound
# =====================================================

BASE_DIR = os.path.expanduser("~/Desktop/PA/sound")
os.makedirs(BASE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"std1_noise_record_{timestamp}.wav"
filepath = os.path.join(BASE_DIR, filename)

print("========================================")
print("WAV will be saved to:")
print(filepath)
print("========================================")

# =====================================================
# 3. Initialize PyAudio
# =====================================================

audio = pyaudio.PyAudio()

# 如果你已经确认 Focusrite 的 device index，
# 可以在这里填写，例如：INPUT_DEVICE_INDEX = 1
INPUT_DEVICE_INDEX = None

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    input_device_index=INPUT_DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

print("Recording started (10 seconds)...")
print("Please generate click sounds now.")

# =====================================================
# 4. Record audio
# =====================================================

frames = []
num_frames = int(SAMPLE_RATE / CHUNK * DURATION)

for _ in range(num_frames):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("Recording finished.")

# =====================================================
# 5. Stop and close stream
# =====================================================

stream.stop_stream()
stream.close()
audio.terminate()

# =====================================================
# 6. Validate data
# =====================================================

raw_bytes = b"".join(frames)

if len(raw_bytes) == 0:
    raise RuntimeError("No audio data captured! Check input device and permissions.")

# =====================================================
# 7. Save WAV file
# =====================================================

with wave.open(filepath, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(raw_bytes)

print("========================================")
print("WAV file saved successfully!")
print(filepath)
print("File size (bytes):", os.path.getsize(filepath))
print("========================================")
