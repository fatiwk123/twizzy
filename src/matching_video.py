import cv2
import numpy as np
import os

# ========== Prétraitement (grayscale + resize) ==========
def preprocess_image(image, target_size=(100, 100)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized

# ========== Charger les templates avec ORB déjà calculé ==========
def load_templates(templates_dir, target_size=(100, 100)):
    orb = cv2.ORB_create()
    templates = {}
    for file in os.listdir(templates_dir):
        if file.endswith((".jpg", ".png")):
            path = os.path.join(templates_dir, file)
            img = cv2.imread(path)
            if img is not None:
                processed = preprocess_image(img, target_size)
                kp, des = orb.detectAndCompute(processed, None)
                if des is not None:
                    templates[file] = (processed, kp, des)
    return templates

# ========== Détection des zones rouges ==========
def detect_red_regions(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 50, 20])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ========== Matching d'une ROI avec les templates ==========
def match_roi(roi, templates, orb, bf, target_size=(100, 100), seuil_distance=200):
    processed = preprocess_image(roi, target_size)
    kp_roi, des_roi = orb.detectAndCompute(processed, None)
    if des_roi is None:
        return None, 0

    best_match_count = 0
    best_name = None

    for name, (tpl_img, kp_tpl, des_tpl) in templates.items():
        matches = bf.match(des_roi, des_tpl)
        bonnes_matches = [m for m in matches if m.distance < seuil_distance]
        if len(bonnes_matches) > best_match_count:
            best_match_count = len(bonnes_matches)
            best_name = name

    return best_name, best_match_count


def match_video(video_path, templates_dir = 'templates'):
    SEUIL_CORRESPONDANCES = 5  # Seuil pour valider un match

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    templates = load_templates(templates_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la vidéo.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original = frame.copy()
        contours = detect_red_regions(frame)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = original[y:y+h, x:x+w]

            if roi.size == 0 or w < 20 or h < 20:
                continue  # Ignore les petits bruits

            best_name, match_count = match_roi(roi, templates, orb, bf)

            if best_name and match_count >= SEUIL_CORRESPONDANCES:
                label = f"{best_name} ({match_count})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        cv2.imshow("Vidéo - Détection et Reconnaissance", frame)
        key = cv2.waitKey(100)  # 30ms entre chaque frame (~30 FPS)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# video_path = 'video/video1.avi'  # Mets ici le chemin vers ta vidéo
# templates_dir = 'templates'
# match_video(video_path)