import cv2 as cv
import numpy as np
import argparse

def hitung_tinggi_badan(pixel_objek, pixel_total, tinggi_kamera, jarak_kamera_ke_objek):
    tinggi_objek = (pixel_objek / pixel_total) * (tinggi_kamera + jarak_kamera_ke_objek)
    # print(f"pixel_objek: {pixel_objek}, pixel_total: {pixel_total}, tinggi_kamera: {tinggi_kamera}, jarak_kamera_ke_objek: {jarak_kamera_ke_objek}")

    return tinggi_objek

def hitung_berat_badan_ideal(tinggi_badan_cm, jenis_kelamin):
    if jenis_kelamin.lower() == 'pria':
        berat_badan_ideal = (tinggi_badan_cm - 100) - ((tinggi_badan_cm - 100) * 0.10)
    elif jenis_kelamin.lower() == 'wanita':
        berat_badan_ideal = (tinggi_badan_cm - 100) - ((tinggi_badan_cm - 100) * 0.15)
    else:
        raise ValueError("Jenis kelamin tidak valid. Harap masukkan 'pria' atau 'wanita'.")
    return berat_badan_ideal

# Argument parser untuk pengaturan aplikasi
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
parser.add_argument('--jenis_kelamin', required=True, choices=['pria', 'wanita'], help='Jenis kelamin (pria atau wanita)')

args = parser.parse_args()

berat_badan = 0.0

# Daftar posisi tubuh dan hubungan antara posisi-posisi tersebut
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
             "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
             "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
             "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# Konfigurasi input untuk model OpenPose
inWidth = args.width
inHeight = args.height

# Membaca model OpenPose dari file TensorFlow
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# Konstanta kalibrasi
tinggi_kamera = 1  # Tinggi kamera dari tanah dalam meter
jarak_kamera_ke_objek = 3  # Jarak antara kamera dan objek dalam meter

# Membuka video dari kamera atau file video
cap = cv.VideoCapture(args.input if args.input else 0)

while True:
    # Membaca frame dari video
    hasFrame, frame = cap.read()
    if not hasFrame:
        break

    # Mendapatkan dimensi frame
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # print(frameWidth, frameHeight)

    # Menyiapkan frame sebagai input untuk model OpenPose
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    # Memastikan jumlah posisi tubuh sesuai dengan yang diharapkan
    assert(len(BODY_PARTS) == out.shape[1])

    # Mendeteksi posisi tubuh dalam frame
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > args.thr else None)

    # Menggambar garis dan elips berdasarkan posisi tubuh
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # Menghitung tinggi badan dari bagian atas kepala ke bagian bawah kaki
    if points[BODY_PARTS["Neck"]] and points[BODY_PARTS["LAnkle"]]:
        tinggi_badan_pixel = points[BODY_PARTS["LAnkle"]][1] - points[BODY_PARTS["Neck"]][1]
        tinggi_badan_m = hitung_tinggi_badan(tinggi_badan_pixel, frameHeight, tinggi_kamera, jarak_kamera_ke_objek)
        print(f'Tinggi badan: {tinggi_badan_m}')
        tinggi_badan_cm = tinggi_badan_m * 100
        print(f'Tinggi badan: {tinggi_badan_cm}')
        berat_badan_ideal = hitung_berat_badan_ideal((tinggi_badan_cm), args.jenis_kelamin)
        cv.putText(frame, f'Tinggi: {tinggi_badan_m:.2f} cm, Berat Ideal: {berat_badan_ideal:.2f} kg', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Menampilkan waktu eksekusi model OpenPose di layar
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Menampilkan frame yang telah diproses
    cv.imshow('OpenPose using OpenCV', frame)

    # Tombol close: Tekan 'q' atau tombol Escape (27) untuk keluar
    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Setelah keluar dari loop, lepaskan sumber daya
cap.release()
cv.destroyAllWindows()
