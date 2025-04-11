# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# import librosa
# import soundfile as sf
# from deepface import DeepFace
# import os

# app = Flask(__name__)

# # Function to analyze facial expression
# def analyze_facial_expression(image_path):
#     try:
#         analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'])
#         return analysis[0]['dominant_emotion']
#     except Exception as e:
#         return str(e)

# # Function to analyze audio emotion
# def analyze_audio(audio_path):
#     try:
#         y, sr = librosa.load(audio_path)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         avg_mfcc = np.mean(mfccs, axis=1)
        
#         # Simple rule-based classification (replace with ML model)
#         if avg_mfcc[0] < -50:
#             return "sad"
#         elif avg_mfcc[0] > 50:
#             return "happy"
#         else:
#             return "neutral"
#     except Exception as e:
#         return str(e)

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     if 'video' in request.files:
#         video = request.files['video']
#         video_path = "temp_video.mp4"
#         video.save(video_path)

#         cap = cv2.VideoCapture(video_path)
#         ret, frame = cap.read()
#         if ret:
#             image_path = "temp_image.jpg"
#             cv2.imwrite(image_path, frame)
#             emotion = analyze_facial_expression(image_path)
#             cap.release()
#             os.remove(image_path)
#             os.remove(video_path)
#             return jsonify({"facial_emotion": emotion})
    
#     if 'audio' in request.files:
#         audio = request.files['audio']
#         audio_path = "temp_audio.wav"
#         audio.save(audio_path)
#         emotion = analyze_audio(audio_path)
#         os.remove(audio_path)
#         return jsonify({"audio_emotion": emotion})

#     return jsonify({"error": "No valid input received"}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify
import cv2
import os
from flask_cors import CORS
import tempfile
from deepface import DeepFace
from collections import Counter

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes
CORS(app, origins=["http://localhost:3000"])

def detect_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_emotions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze every 5th frame to balance speed and accuracy
        if frame_count % 5 == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # DeepFace might return a list or dict depending on version
                emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                detected_emotions.append(emotion)
                print(f"Frame {frame_count} ➤ Detected Emotion: {emotion}")
            except Exception as e:
                print(f"Frame {frame_count} ➤ Error: {str(e)}")

        frame_count += 1

    cap.release()

    if detected_emotions:
        # Count all detected emotions and return the most frequent one
        most_common = Counter(detected_emotions).most_common(1)[0][0]
        print(f"✅ Final Detected Emotion: {most_common}")
        return most_common
    else:
        return "No face/emotion detected"

@app.route("/detect-emotion", methods=["POST"])
def detect_emotion_api():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.webm")
    video.save(temp_video_path)

    try:
        final_emotion = detect_emotion(temp_video_path)
        os.remove(temp_video_path)
        print("final_emotion",final_emotion)
        return jsonify(final_emotion), 200
    except Exception as e:
        print("❌ Detection Error:", str(e))
        return jsonify({"error": "Emotion detection failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
