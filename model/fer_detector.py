from deepface import DeepFace
import cv2

def detect_emotion(frame):
    try:
        # DeepFace butuh RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)

        # Ambil skor emosi penting
        emotions = result[0]['emotion']

        score = 0
        score += int(emotions['fear'] * 0.3)
        score += int(emotions['angry'] * 0.3)
        score += int(emotions['sad'] * 0.2)
        score += int(emotions['disgust'] * 0.2)

        # Tentukan status akhir
        if score < 4:
            status = "Tenang"
        elif score < 7:
            status = "Cemas Ringan"
        elif score < 10:
            status = "Cemas Sedang"
        else:
            status = "Stres Tinggi"

        return int(score), status
    except:
        return 0, "Wajah Tidak Terdeteksi"
