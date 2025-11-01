# src/transcriber.py
import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Choose a base model for inference. You can replace with a fine-tuned model later.
MODEL_NAME = "facebook/wav2vec2-base-960h"

# Load processor and model (downloads from Hugging Face the first time)
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

TARGET_SR = 16000

def load_audio(path, target_sr=TARGET_SR):
    wave, sr = torchaudio.load(path)
    # mono
    if wave.shape[0] > 1:
        wave = torch.mean(wave, dim=0, keepdim=True)
    wave = wave.squeeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wave = resampler(wave)
    # normalize to -1..1
    wave = wave / torch.abs(wave).max()
    return wave.numpy(), target_sr

def transcribe_file(path):
    speech, sr = load_audio(path)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

def main(input_folder="data/sample_audio", output_folder="data/transcripts"):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".wav")]
    if not files:
        print("No .wav files found in", input_folder)
        return
    for f in files:
        path = os.path.join(input_folder, f)
        print("Transcribing:", path)
        text = transcribe_file(path)
        out_path = os.path.join(output_folder, f.replace(".wav", ".txt"))
        with open(out_path, "w") as wf:
            wf.write(text)
        print("Saved:", out_path)
        print("Transcript (first 200 chars):", text[:200])

if __name__ == "__main__":
    main()
