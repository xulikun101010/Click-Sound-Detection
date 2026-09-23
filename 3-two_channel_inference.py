import os
import json
import wave
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


# ============================================================
# Paths
# ============================================================

LATEST_JSON_PATH = r"D:\Acoustics\data\recordings\latest_recording.json"

MODEL_DIR = r"D:\Acoustics\EConMateSense\models"

AIR_STATE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_state_air_model.pth")
STRUCT_STATE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_state_structure_model.pth")

AIR_PLUG_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_noise_air_model.pth")
STRUCT_PLUG_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_noise_structure_model.pth")

OUTPUT_DIR = r"D:\Acoustics\data\inference_outputs"
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "segments")
SPECTROGRAM_DIR = os.path.join(OUTPUT_DIR, "spectrograms")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")


# ============================================================
# Classes
# ============================================================

STATE_CLASSES = ["successful", "failed"]
PLUG_CLASSES = ["Bedkom", "LSRegler", "SchaltBedb"]


# ============================================================
# Signal / Spectrogram Parameters
# ============================================================

SEGMENT_DURATION = 0.2
PRE_TIME = 0.10
POST_TIME = 0.10

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 0
F_MAX = None
TOP_DB = 80


# ============================================================
# CNN Model
# ============================================================

class CNNModel(nn.Module):
    """
    CNN architecture used for both:
    1. State classification: successful / failed
    2. Plug type classification: Bedkom / LSRegler / SchaltBedb

    Air input shape:
        [batch, 1, 128, 76]

    Structure input shape:
        [batch, 1, 128, 19]
    """

    def __init__(self, input_width, num_classes):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Height: 128 -> 64 -> 32 -> 16
        # Air width: 76 -> 38 -> 19 -> 9
        # Structure width: 19 -> 9 -> 4 -> 2
        width_after_pooling = input_width // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * width_after_pooling, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Utility Functions
# ============================================================

def ensure_output_dirs():
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_latest_recording(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"latest_recording.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    air_path = info["air"]["path"]
    struct_path = info["structure"]["path"]

    if not os.path.exists(air_path):
        raise FileNotFoundError(f"Air wav not found: {air_path}")

    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"Structure wav not found: {struct_path}")

    return info, air_path, struct_path


def load_wav_mono(wav_path):
    """
    Load wav file with original sampling rate.
    """
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    return audio, sr


def extract_peak_segment(audio, sr, pre_time=0.10, post_time=0.10):
    """
    Extract 0.2 s segment around the maximum absolute amplitude.

    This version always returns a fixed-length segment.
    If the peak is too close to the beginning or end, zero-padding is applied.
    """

    target_len = int((pre_time + post_time) * sr)

    peak_idx = int(np.argmax(np.abs(audio)))

    start = peak_idx - int(pre_time * sr)
    end = peak_idx + int(post_time * sr)

    pad_left = 0
    pad_right = 0

    if start < 0:
        pad_left = -start
        start = 0

    if end > len(audio):
        pad_right = end - len(audio)
        end = len(audio)

    segment = audio[start:end]

    if pad_left > 0 or pad_right > 0:
        segment = np.pad(segment, (pad_left, pad_right), mode="constant")

    if len(segment) > target_len:
        segment = segment[:target_len]

    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)), mode="constant")

    return segment, peak_idx / sr


def save_segment(segment, sr, output_path):
    sf.write(output_path, segment, sr)


def generate_mel_spectrogram(segment, sr):
    """
    Generate Mel-spectrogram in dB scale.
    Same logic as the previous training preprocessing.
    """

    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel_spec,
        ref=np.max,
        top_db=TOP_DB
    )

    return mel_db.astype(np.float32)


def normalize_spectrogram(spec):
    """
    Normalize dB spectrogram from approximately [-80, 0] to [0, 1].
    """
    spec = (spec + 80.0) / 80.0
    spec = np.clip(spec, 0.0, 1.0)
    return spec


def prepare_tensor(spec, expected_shape):
    """
    Convert spectrogram to CNN input tensor.
    """

    if spec.shape != expected_shape:
        raise ValueError(
            f"Unexpected spectrogram shape: {spec.shape}, expected: {expected_shape}"
        )

    spec = normalize_spectrogram(spec)

    # [H, W] -> [1, 1, H, W]
    spec = np.expand_dims(spec, axis=0)
    spec = np.expand_dims(spec, axis=0)

    tensor = torch.tensor(spec, dtype=torch.float32)
    return tensor


def load_model(model_path, input_width, num_classes, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = CNNModel(input_width=input_width, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def predict(model, input_tensor, class_names, device):
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)

        confidence, pred_idx = torch.max(probabilities, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()

    pred_class = class_names[pred_idx]

    prob_dict = {
        class_names[i]: float(probabilities[0, i].item())
        for i in range(len(class_names))
    }

    return pred_class, confidence, prob_dict


def process_channel(
    channel_name,
    wav_path,
    expected_shape,
    state_model,
    plug_model,
    device
):
    """
    Complete inference for one channel:
    wav -> segment -> spectrogram -> state CNN -> plug CNN if successful
    """

    print(f"\n==============================")
    print(f"{channel_name.upper()} CHANNEL")
    print(f"==============================")
    print(f"WAV path: {wav_path}")

    audio, sr = load_wav_mono(wav_path)

    segment, peak_time = extract_peak_segment(
        audio=audio,
        sr=sr,
        pre_time=PRE_TIME,
        post_time=POST_TIME
    )

    base_name = os.path.splitext(os.path.basename(wav_path))[0]

    segment_path = os.path.join(
        SEGMENT_DIR,
        f"{base_name}_segment_0p2s.wav"
    )

    spec_path = os.path.join(
        SPECTROGRAM_DIR,
        f"{base_name}_mel.npy"
    )

    save_segment(segment, sr, segment_path)

    spec = generate_mel_spectrogram(segment, sr)
    np.save(spec_path, spec)

    print(f"Sampling rate: {sr}")
    print(f"Peak time: {peak_time:.4f} s")
    print(f"Segment saved: {segment_path}")
    print(f"Spectrogram shape: {spec.shape}")
    print(f"Spectrogram saved: {spec_path}")

    input_tensor = prepare_tensor(spec, expected_shape)

    # ------------------------------
    # Stage 1: State prediction
    # ------------------------------
    state_pred, state_conf, state_probs = predict(
        model=state_model,
        input_tensor=input_tensor,
        class_names=STATE_CLASSES,
        device=device
    )

    print(f"\nState prediction: {state_pred}")
    print(f"State confidence: {state_conf:.4f}")
    print(f"State probabilities: {state_probs}")

    # ------------------------------
    # Stage 2: Plug type prediction
    # ------------------------------
    if state_pred == "successful":
        plug_pred, plug_conf, plug_probs = predict(
            model=plug_model,
            input_tensor=input_tensor,
            class_names=PLUG_CLASSES,
            device=device
        )

        print(f"\nPlug type prediction: {plug_pred}")
        print(f"Plug type confidence: {plug_conf:.4f}")
        print(f"Plug type probabilities: {plug_probs}")

    else:
        plug_pred = "skipped"
        plug_conf = 0.0
        plug_probs = {}

        print("\nPlug type prediction skipped because state = failed.")

    result = {
        "channel": channel_name,
        "wav_path": wav_path,
        "sampling_rate": sr,
        "peak_time_sec": peak_time,
        "segment_path": segment_path,
        "spectrogram_path": spec_path,
        "spectrogram_shape": list(spec.shape),
        "state_prediction": state_pred,
        "state_confidence": state_conf,
        "state_probabilities": state_probs,
        "plug_type_prediction": plug_pred,
        "plug_type_confidence": plug_conf,
        "plug_type_probabilities": plug_probs
    }

    return result


def fuse_results(air_result, struct_result):
    """
    Conservative fusion rule.
    """

    air_state = air_result["state_prediction"]
    struct_state = struct_result["state_prediction"]

    air_plug = air_result["plug_type_prediction"]
    struct_plug = struct_result["plug_type_prediction"]

    # Case 1: both successful
    if air_state == "successful" and struct_state == "successful":
        final_state = "successful"

        if air_plug == struct_plug:
            final_plug_type = air_plug
            final_comment = "Both channels agree on successful insertion and plug type."
        else:
            final_plug_type = "uncertain"
            final_comment = "Both channels detect successful insertion, but plug type predictions are different."

    # Case 2: both failed
    elif air_state == "failed" and struct_state == "failed":
        final_state = "failed"
        final_plug_type = "skipped"
        final_comment = "Both channels detect failed insertion."

    # Case 3: disagreement
    else:
        final_state = "uncertain"
        final_plug_type = "uncertain"
        final_comment = "Air and structure channels disagree on insertion state."

    return {
        "final_state": final_state,
        "final_plug_type": final_plug_type,
        "comment": final_comment
    }


def save_result_json(recording_info, air_result, struct_result, final_result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_path = os.path.join(
        RESULT_DIR,
        f"inference_result_{timestamp}.json"
    )

    all_results = {
        "recording_info": recording_info,
        "air_result": air_result,
        "structure_result": struct_result,
        "final_result": final_result
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"\nResult saved to: {result_path}")


# ============================================================
# Main
# ============================================================

def main():
    ensure_output_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    recording_info, air_wav_path, struct_wav_path = load_latest_recording(
        LATEST_JSON_PATH
    )

    print("\nLatest recording loaded:")
    print("Air wav:", air_wav_path)
    print("Structure wav:", struct_wav_path)

    # Load models
    air_state_model = load_model(
        AIR_STATE_MODEL_PATH,
        input_width=76,
        num_classes=2,
        device=device
    )

    struct_state_model = load_model(
        STRUCT_STATE_MODEL_PATH,
        input_width=19,
        num_classes=2,
        device=device
    )

    air_plug_model = load_model(
        AIR_PLUG_MODEL_PATH,
        input_width=76,
        num_classes=3,
        device=device
    )

    struct_plug_model = load_model(
        STRUCT_PLUG_MODEL_PATH,
        input_width=19,
        num_classes=3,
        device=device
    )

    print("\nAll CNN models loaded successfully.")

    # Process air channel
    air_result = process_channel(
        channel_name="air",
        wav_path=air_wav_path,
        expected_shape=(128, 76),
        state_model=air_state_model,
        plug_model=air_plug_model,
        device=device
    )

    # Process structure channel
    struct_result = process_channel(
        channel_name="structure",
        wav_path=struct_wav_path,
        expected_shape=(128, 19),
        state_model=struct_state_model,
        plug_model=struct_plug_model,
        device=device
    )

    # Final decision
    final_result = fuse_results(air_result, struct_result)

    print("\n==============================")
    print("FINAL DECISION")
    print("==============================")
    print(f"Final state: {final_result['final_state']}")
    print(f"Final plug type: {final_result['final_plug_type']}")
    print(f"Comment: {final_result['comment']}")

    save_result_json(
        recording_info=recording_info,
        air_result=air_result,
        struct_result=struct_result,
        final_result=final_result
    )


if __name__ == "__main__":
    main()