import os
import re
import wave
import threading
import math
import pyaudio

RECORD_SECONDS = 5
FORMAT = pyaudio.paInt16
CHANNELS = 1

# DEVICE 1 (contact sensor)
DEVICE_1 = 1
SR_1 = 48000
CHUNK_1 = 1024
LABEL_1 = "STRUCT"
OUT1_SUFFIX = "struct"

# DEVICE 2 (Dodotronic Ultramic 384k)
DEVICE_2 = 2
SR_2 = 192000
CHUNK_2 = 1024
LABEL_2 = "AIR"
OUT2_SUFFIX = "air"

BASE_DIR = os.path.join("data", "recordings")
AIR_DIR = os.path.join(BASE_DIR, "airborne")
STRUCT_DIR = os.path.join(BASE_DIR, "structure_borne")
INDEX_WIDTH = 3

audio1 = pyaudio.PyAudio()
audio2 = pyaudio.PyAudio()

def list_audio_devices(audio = None):
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']}")

def extract_index_from_filename(fname: str):
    stem, _ = os.path.splitext(fname)
    m = re.match(r"^(?:recording_)?(\d+)_", stem)
    if not m:
        return None
    return int(m.group(1))

def next_recording_index(folders):
    max_idx = 0
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".wav"):
                idx = extract_index_from_filename(fname)
                if idx is not None:
                    max_idx = max(max_idx, idx)
    return max_idx + 1

def record_device(device_index, sampling_rate, chunksize, pyaudio_instance, output_filename, start_event, label):
    try:
        
        print(f"[{label}] Opening stream on device {device_index}...")
        stream = pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sampling_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunksize
        )

        frames = []
        num_chunks = int(sampling_rate / chunksize * RECORD_SECONDS)

        start_event.wait()

        print(f"[{label}] Recording started...")
        for _ in range(num_chunks):
            data = stream.read(chunksize)
            frames.append(data)

        sample_width = pyaudio_instance.get_sample_size(FORMAT)

        stream.stop_stream()
        stream.close()

        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(sampling_rate)
            wf.writeframes(b"".join(frames))

        print(f"[{label}] Saved: {output_filename}")

    except Exception as e:
        print(f"[{label}] Error: {e}")

def main():
    os.makedirs(AIR_DIR, exist_ok=True)
    os.makedirs(STRUCT_DIR, exist_ok=True)

    start_event = threading.Event()

    idx = next_recording_index([AIR_DIR, STRUCT_DIR])
    idx_str = str(idx).zfill(INDEX_WIDTH)

    out_1 = os.path.join(STRUCT_DIR, f"{idx_str}_{OUT1_SUFFIX}_Bedkom_Low.wav")
    out_2 = os.path.join(AIR_DIR, f"{idx_str}_{OUT2_SUFFIX}_Bedkom_Low.wav")

    thread1 = threading.Thread(
        target=record_device,
        args=(DEVICE_1, SR_1, CHUNK_1, audio1, out_1, start_event, LABEL_1)
    )
    thread2 = threading.Thread(
        target=record_device,
        args=(DEVICE_2, SR_2, CHUNK_2, audio2, out_2, start_event, LABEL_2)
    )

    thread1.start()
    thread2.start()

    print("Both streams are ready. Starting recording...")
    start_event.set()

    thread1.join()
    thread2.join()

    audio1.terminate()
    audio2.terminate()

    print("Done.")

if __name__ == "__main__":
    main()