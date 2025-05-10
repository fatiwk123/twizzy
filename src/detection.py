import cv2
import numpy as np

# 1. Charger une image d'exemple
image = cv2.imread('twizzy/images/image_open_CV/circles.jpg')


# 2. Vérifier le chargement
if image is None:
    print("Erreur : image non chargée")
    exit()

# 3. Redimensionner (optionnel)
image = cv2.resize(image, (640, 480))

# 4. Convertir en HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 5. Définir la plage pour détecter le rouge
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 6. Créer les masques pour le rouge
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# 7. Appliquer le masque
red_only = cv2.bitwise_and(image, image, mask=red_mask)

# 8. Afficher les résultats
cv2.imshow("Image originale", image)
cv2.imshow("Seuillage rouge", red_mask)
cv2.imshow("Parties rouges", red_only)
# 9. Convertir le masque rouge en niveaux de gris pour la détection de contours
gray_mask = red_mask.copy()

# 10. Détecter les contours
contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 11. Parcourir tous les contours
for cnt in contours:
    
    area = cv2.contourArea(cnt)
    if area < 300:
        continue  # ignorer petits bruits

    # Approximation de la forme
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

    # Boîte autour
    x, y, w, h = cv2.boundingRect(approx)

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
    
    # Dessiner le contour
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Écrire le nom
    cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# 12. Afficher image finale avec formes détectées
cv2.imshow("Formes detectees", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()
