import cv2
import csv
import time
from datetime import datetime
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

# Buka webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Gunakan DirectShow untuk Windows

# Untuk menghitung FPS
prev_time = time.time()

# Variabel untuk menghitung ekspresi negatif berturut-turut
negative_counter = 0
timestamps = []  # Menyimpan waktu untuk grafik
emotions = []  # Menyimpan emosi untuk grafik
diagnoses = []  # Menyimpan hasil diagnosis

# Fungsi untuk menganalisis ekspresi wajah menggunakan DeepFace
def analyze_expression(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion'], result[0]['region']
    except:
        return "No Face", None

# Fungsi untuk diagnosis berdasarkan emosi
def get_diagnosis(emotion):
    if emotion in ['sad', 'angry', 'fear']:
        return "⚠️ Possible Depression or Anxiety"
    else:
        return "Normal"

# Baca satu frame untuk dapatkan ukuran
ret, frame = cap.read()
if not ret:
    print("❌ Tidak dapat membuka webcam.")
    exit()

frame_height, frame_width = frame.shape[:2]

# Mulai webcam virtual
with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=20, fmt=PixelFormat.BGR) as cam:
    print(f"🎥 Virtual camera aktif: {cam.device}")
    print("Tekan 'q' atau 'ESC' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        emotion, face_region = analyze_expression(frame)

        if emotion in ['sad', 'angry', 'fear']:
            negative_counter += 1
        else:
            negative_counter = 0

        if negative_counter > 500:
            suspected_disorder = "⚠️ Possible signs of depression or anxiety"
            diagnosis = get_diagnosis(emotion)
            cv2.putText(frame, suspected_disorder, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            diagnosis = get_diagnosis(emotion)

        diagnoses.append(diagnosis)

        if face_region:
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f"Emotion: {emotion}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {timestamp}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.putText(frame, f"Diagnosis: {diagnosis}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Simpan ke CSV
        with open('emotion_log.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, emotion, diagnosis])

        timestamps.append(timestamp)
        emotions.append(emotion)

        # Tampilkan ke layar
        cv2.imshow("Emotion Detection", frame)

        # Kirim ke virtual cam (Zoom)
        cam.send(frame)
        cam.sleep_until_next_frame()

        # Cek tombol keluar
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # ESC
            break

# Release kamera
cap.release()
cv2.destroyAllWindows()

# Visualisasi setelah program selesai
unique_emotions = list(set(emotions))
emotion_to_num = {emotion: i for i, emotion in enumerate(unique_emotions)}
emotion_values = [emotion_to_num[e] for e in emotions]

plt.figure(figsize=(12, 6))
plt.plot(timestamps, emotion_values, marker='o')
plt.xticks(rotation=45)
plt.yticks(ticks=range(len(unique_emotions)), labels=unique_emotions)
plt.title("Emosi Dominan dalam Waktu Tertentu")
plt.xlabel("Waktu")
plt.ylabel("Emosi")
plt.grid(True)
plt.tight_layout()
plt.show()
