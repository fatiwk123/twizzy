import cv2
import numpy as np
import os

# === PARAMÈTRES ===
IMAGE_PATH = 'images/image_test__final/G0020129.JPG'
TEMPLATES_DIR = 'templates'
OUTPUT_DIR = 'panneaux_extraits'
TARGET_SIZE = (100, 100)
SEUIL_BONNES_CORRESPONDANCES = 10

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def preprocess_image(image, target_size=TARGET_SIZE):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized

def load_templates(templates_dir):
    templates = {}
    for file in os.listdir(templates_dir):
        if file.endswith((".jpg", ".png")):
            path = os.path.join(templates_dir, file)
            img = cv2.imread(path)
            if img is not None:
                processed = preprocess_image(img)
                kp, des = orb.detectAndCompute(processed, None)
                if des is not None:
                    templates[file] = (processed, kp, des)
    return templates

def detect_and_recognize(image_path, templates, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : impossible de charger l'image {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_clean = image.copy()
    image = cv2.resize(image, (640, 480))
    image_clean = cv2.resize(image_clean, (640, 480))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # === Étape 1: Masque rouge ===
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

    cv2.imshow("Étape 1 - Masque Rouge", red_mask)
    cv2.waitKey(0)

    # === Étape 2: Contours ===
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 2)
    cv2.imshow("Étape 2 - Contours Détectés", image_with_contours)
    cv2.waitKey(0)

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        shape = "Inconnu"
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) > 6:
            shape = "Cercle"

        x, y, w, h = cv2.boundingRect(cnt)
        roi = image_clean[y:y+h, x:x+w]

        if roi.size == 0 or w < 20 or h < 20:
            continue

        # === Étape 3: Affichage du ROI ===
        cv2.imshow("Étape 3 - ROI", roi)
        cv2.waitKey(0)

        # === Étape 4: Matching ORB ===
        processed = preprocess_image(roi)
        kp_roi, des_roi = orb.detectAndCompute(processed, None)
        if des_roi is None:
            continue

        best_match_name = None
        best_match_count = 0
        best_matches = []
        best_kp_tpl = []

        for name, (tpl_img, kp_tpl, des_tpl) in templates.items():
            matches = bf.match(des_roi, des_tpl)
            good_matches = [m for m in matches if m.distance < 64]
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_match_name = name
                best_matches = good_matches
                best_kp_tpl = kp_tpl

        if best_match_name and best_match_count >= SEUIL_BONNES_CORRESPONDANCES:
            label = f"{best_match_name} ({best_match_count})"
            print(f"[MATCH] {label}")

            # Affichage du matching
            tpl_img, kp_tpl, _ = templates[best_match_name]
            result = cv2.drawMatches(processed, kp_roi, tpl_img, best_kp_tpl, best_matches[:20], None, flags=2)
            cv2.putText(result, f"Match: {best_match_name}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Étape 4 - Matching ORB", result)
            cv2.waitKey(0)

            # === Étape 5: Dessin final sur l’image ===
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            filename = os.path.join(output_dir, f"{best_match_name}_{x}_{y}.jpg")
            cv2.imwrite(filename, roi)

    # === Étape finale: Image annotée ===
    cv2.imshow("Étape Finale - Résultat Global", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === LANCEMENT ===
if __name__ == "__main__":
    templates = load_templates(TEMPLATES_DIR)
    detect_and_recognize(IMAGE_PATH, templates, OUTPUT_DIR)
