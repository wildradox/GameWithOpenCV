"""
Deteksi Pose "2 Jari" dengan OpenCV + MediaPipe
------------------------------------------------
- Kamera dibuka dengan backend MSMF terlebih dahulu, kalau gagal fallback ke DSHOW.
- Ketika tangan menunjukkan pose "2" (telunjuk + jari tengah terangkat,
  ibu jari/jari manis/kelingking turun), frame kamera akan di-blur.

Instalasi:
    pip install opencv-python mediapipe

Jalankan:
    python deteksi_pose_dua.py

Tekan 'q' untuk keluar.
"""

import cv2
import mediapipe as mp


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Coba buka kamera dengan MSMF dulu, baru fallback ke DSHOW."""
    backends = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
    ]

    for backend_id, backend_name in backends:
        cap = cv2.VideoCapture(index, backend_id)
        if cap.isOpened():
            print(f"[INFO] Kamera berhasil dibuka dengan backend: {backend_name}")
            return cap
        cap.release()

    raise RuntimeError(
        "Tidak bisa membuka kamera dengan MSMF maupun DSHOW. "
        "Cek apakah kamera sedang dipakai aplikasi lain atau index kamera salah."
    )


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, handedness_label: str) -> list[int]:
    """
    Mengembalikan list 5 nilai [ibu_jari, telunjuk, tengah, manis, kelingking]
    berisi 1 (terangkat) atau 0 (turun), berdasarkan landmark MediaPipe.
    """
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Ibu jari: dicek berdasarkan posisi x (arahnya horizontal),
    # bedakan tangan kanan/kiri karena arahnya berlawanan.
    thumb_tip = hand_landmarks.landmark[tip_ids[0]]
    thumb_ip = hand_landmarks.landmark[tip_ids[0] - 1]
    if handedness_label == "Right":
        fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)
    else:
        fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)

    # 4 jari lainnya: tip lebih tinggi (y lebih kecil) dari sendi PIP -> terangkat
    for tip_id in tip_ids[1:]:
        tip_y = hand_landmarks.landmark[tip_id].y
        pip_y = hand_landmarks.landmark[tip_id - 2].y
        fingers.append(1 if tip_y < pip_y else 0)

    return fingers


def is_pose_two(fingers: list[int]) -> bool:
    """Pose '2': telunjuk & tengah naik, ibu jari/manis/kelingking turun."""
    thumb, index, middle, ring, pinky = fingers
    return index == 1 and middle == 1 and ring == 0 and pinky == 0


def main():
    cap = open_camera(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARNING] Gagal membaca frame dari kamera, berhenti.")
                break

            frame = cv2.flip(frame, 1)  # efek cermin biar lebih natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            pose_detected = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" / "Right"
                    fingers = count_fingers(hand_landmarks, label)

                    if is_pose_two(fingers):
                        pose_detected = True

                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            if pose_detected:
                frame = cv2.GaussianBlur(frame, (35, 35), 0)
                cv2.putText(
                    frame, "POSE 2 TERDETEKSI - BLUR AKTIF", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
            else:
                cv2.putText(
                    frame, "Tunjukkan pose 2 jari (telunjuk + tengah)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            cv2.imshow("Deteksi Pose 2 - OpenCV", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
