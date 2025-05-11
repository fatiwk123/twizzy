import cv2
import numpy as np
import os

# === 1. Charger l'image de test ===
test_image = cv2.imread("images/image_test__final/GOPR0156.jpg")
test_image = cv2.resize(test_image, (640, 480))
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# === 2. Détection des contours ===
edges = cv2.Canny(gray, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === 3. Extraire les zones candidates ===
candidates = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 50 and h > 50:  # Ignorer les petits objets
        crop = test_image[y:y+h, x:x+w]
        candidates.append((crop, (x, y, w, h)))

# === 4. Initialiser ORB et templates ===
orb = cv2.ORB_create()
template_folder = "templates"
template_labels = {
    "90.png": "Limite 90",
    "70.png": "Limite 70",
    "stop.png": "Stop",
    "deux_voies.png": "Deux Voies"
}
template_images = {
    name: cv2.imread(os.path.join(template_folder, name), 0)
    for name in template_labels
}

# === 5. Comparer chaque zone candidate à chaque template ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for crop, (x, y, w, h) in candidates:
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(crop_gray, None)
    
    best_score = 0
    best_label = None

    for name, template in template_images.items():
        kp2, des2 = orb.detectAndCompute(template, None)
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_label = template_labels[name]

    if best_label:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(test_image, best_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2)

# === 6. Afficher résultat ===
cv2.imshow("Panneaux détectés", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
