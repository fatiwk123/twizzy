import cv2
import numpy as np
import os

# Créer le dossier de sortie s'il n'existe pas
os.makedirs("panneaux_extraits", exist_ok=True)

# 1. Charger une image d'exemple
image = cv2.imread('images/image_test__final/GOPR0143.jpg')

# 2. Vérifier le chargement
if image is None:
    print("Erreur : image non chargée")
    exit()

# 3. Redimensionner (optionnel)
image = cv2.resize(image, (640, 480))

# 4. Convertir en HSV + flou pour lisser les couleurs
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

# 5. Utiliser les seuils que tu as trouvés
# Plage 1 : rouge clair
lower_red1 = np.array([0, 50, 20])
upper_red1 = np.array([12, 255, 255])

# Plage 2 : rouge foncé
lower_red2 = np.array([160, 50, 20])
upper_red2 = np.array([180, 255, 255])

# Masques combinés
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Nettoyage avec morphologie
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((15, 15), np.uint8)  # 🔥 plus grand pour bien combler

red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)


# 7. Appliquer le masque sur l’image
red_only = cv2.bitwise_and(image, image, mask=red_mask)

# 8. Afficher les masques pour contrôle
cv2.imshow("Image originale", image)
cv2.imshow("Masque rouge", red_mask)
cv2.imshow("Parties rouges", red_only)

# 9. Détection des contours
gray_mask = red_mask.copy()
contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 10. Analyse des contours
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    if area < 800 or w < 40 or h < 60:
        continue  # ignorer bruit et petits objets

    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

    # Identifier la forme
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        aspect_ratio = w / float(h)
        shape = "Carre" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
    elif 7 <= len(approx) <= 9:
        shape = "Octogone"
    else:
        shape = "Cercle"

    # Dessiner les contours et le nom (avec ombre blanche)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Extraire la région et sauvegarder
    roi = image[y:y + h, x:x + w]
    filename = f"twizzy/panneaux_extraits/{shape}_{x}_{y}.jpg"
    cv2.imwrite(filename, roi)
    cv2.imshow(f"{shape} extrait", roi)

# 11. Affichage final
cv2.imshow("Formes detectees", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
