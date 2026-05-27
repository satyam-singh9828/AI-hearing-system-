from flask import Flask, render_template, Response, request, send_file
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import os
from pydub import AudioSegment
from gtts import gTTS
from transformers import pipeline

# Speaker detection
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
import librosa
import numpy as np

app = Flask(__name__)

# -------------------------------
# GLOBALS
# -------------------------------
vosk_model = None
summarizer = None
encoder = None

q = queue.Queue()
is_recording = False
frames_buffer = []

os.makedirs("recordings", exist_ok=True)

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    return text.replace("\n", " ").strip()

# -------------------------------
# SUMMARY
# -------------------------------
def generate_summary(text):
    global summarizer

    text = clean_text(text)

    if len(text.split()) < 20:
        return "⚡ Short conversation:\n" + text

    words = text.split()
    chunks = []

    for i in range(0, len(words), 80):
        chunks.append(" ".join(words[i:i+80]))

    final_summary = []

    for chunk in chunks:
        try:
            result = summarizer(
                chunk,
                max_length=60,
                min_length=20,
                do_sample=False
            )

            summary_text = result[0]['summary_text'].strip()

            if not summary_text.endswith("."):
                summary_text += "."

            final_summary.append(summary_text)

        except Exception as e:
            print("❌ Summary error:", e)
            final_summary.append(chunk)

    return "👉 " + " ".join(final_summary)

# -------------------------------
# LIVE SUMMARY API
# -------------------------------
@app.route("/summarize_live", methods=["POST"])
def summarize_live():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return {"summary": "⚠️ Nothing to summarize yet."}

    return {"summary": generate_summary(text)}

# -------------------------------
# SPEAKER DIARIZATION
# -------------------------------
def diarize_lightweight(wav_path):
    global encoder

    wav, sr = librosa.load(wav_path, sr=16000)

    chunk_size = int(sr * 1.5)
    embeddings = []

    for i in range(0, len(wav), chunk_size):
        chunk = wav[i:i+chunk_size]

        if len(chunk) < chunk_size:
            continue

        emb = encoder.embed_utterance(chunk)
        embeddings.append(emb)

    if len(embeddings) == 0:
        return [0]

    embeddings = np.array(embeddings)

    n_speakers = min(3, len(embeddings))
    clustering = AgglomerativeClustering(n_clusters=n_speakers)
    labels = clustering.fit_predict(embeddings)

    return labels

# -------------------------------
# LIVE AUDIO STREAM
# -------------------------------
def live_audio_stream():
    global vosk_model, is_recording, frames_buffer

    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(True)  # 🔥 important

    frames_buffer = []
    is_recording = True

    def callback(indata, frames_count, time, status):
        if is_recording:
            q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        print("🎤 Listening started...")

        while is_recording:
            data = q.get()
            frames_buffer.append(data)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    yield f"data: {text}\n\n"
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "")
                if text:
                    yield f"data: {text}\n\n"

    print("🛑 Listening stopped")

# -------------------------------
# ROUTES
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    return Response(live_audio_stream(), mimetype="text/event-stream")

# -------------------------------
# STOP + SAVE AUDIO
# -------------------------------
@app.route("/stop", methods=["POST"])
def stop():
    global is_recording, frames_buffer

    is_recording = False

    if not frames_buffer:
        return {"status": "no audio"}

    audio_data = b''.join(frames_buffer)

    audio_segment = AudioSegment(
        audio_data,
        sample_width=2,
        frame_rate=16000,
        channels=1
    )

    wav_path = "recordings/recorded.wav"
    audio_segment.export(wav_path, format="wav")

    print("✅ Audio saved after stop")

    return {"status": "saved"}

# -------------------------------
# PROCESS AUDIO (🔥 FIXED)
# -------------------------------
@app.route("/process", methods=["POST"])
def process():
    global vosk_model

    wav_file = "recordings/recorded.wav"

    if not os.path.exists(wav_file):
        return render_template(
            "index.html",
            transcript_plain="❌ No recording",
            transcript_speakers="",
            summary=""
        )

    rec = KaldiRecognizer(vosk_model, 16000)
    rec.SetWords(True)

    full_text = []

    with open(wav_file, "rb") as f:
        while True:
            data = f.read(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if "text" in res:
                    full_text.append(res["text"])

    final_res = json.loads(rec.FinalResult())
    if "text" in final_res:
        full_text.append(final_res["text"])

    text = " ".join(full_text).strip()

    transcript_plain = text

    # Speaker diarization
    labels = diarize_lightweight(wav_file)

    words = text.split()
    chunk_size = 8
    transcript_speakers = ""

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)

        speaker = labels[(i // chunk_size) % len(labels)]
        transcript_speakers += f"Speaker {speaker+1}: {chunk_text}\n"

    summary = generate_summary(text)

    # Save file
    with open("recordings/transcript.txt", "w") as f:
        f.write("Plain Transcript:\n")
        f.write(transcript_plain + "\n\n")
        f.write("Speaker-wise:\n")
        f.write(transcript_speakers)

    return render_template(
        "index.html",
        transcript_plain=transcript_plain,
        transcript_speakers=transcript_speakers,
        summary=summary
    )

# -------------------------------
# DOWNLOAD
# -------------------------------
@app.route("/download")
def download():
    path = "recordings/transcript.txt"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "No file found"

# -------------------------------
# TEXT TO SPEECH
# -------------------------------
@app.route("/reply", methods=["POST"])
def reply():
    user_text = request.form.get("user_text", "")
    audio_path = ""

    if user_text:
        os.makedirs("static", exist_ok=True)
        tts = gTTS(user_text)
        audio_path = "static/reply.mp3"
        tts.save(audio_path)

    return render_template("index.html", reply_audio=audio_path)

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("Loading Vosk...")
    vosk_model = Model("vosk_models/vosk-model-small-en-us-0.15")

    print("Loading summarizer...")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    print("Loading speaker encoder...")
    encoder = VoiceEncoder()

    app.run(debug=True, use_reloader=False)