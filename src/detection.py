import cv2
import numpy as np
import os

def detect_and_extract_shapes(image_path, output_dir = 'panneaux_extraits'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Supprime toutes les images deja dans le dossier de sortie
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete file
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : impossible de charger l'image {image_path}")
        return

    # Créer une copie propre de l'image
    image_clean = image.copy()

    # Redimensionner l'image (optionnel)
    image = cv2.resize(image, (640, 480))
    image_clean = cv2.resize(image_clean, (640, 480))

    # Convertir en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les seuils pour détecter les couleurs spécifiques (exemple : rouge)
    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 50, 20])
    upper_red2 = np.array([180, 255, 255])

    # Créer les masques
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Nettoyer le masque avec des opérations morphologiques
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)

    # Trouver les contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Parcourir les contours détectés
    for cnt in contours:
        # Calculer le périmètre du contour
        perimeter = cv2.arcLength(cnt, True)
        # Approximer le contour pour identifier la forme
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        # Identifier la forme en fonction du nombre de sommets
        shape = "Inconnu"
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) > 6:
            shape = "Cercle"

        # Calculer les coordonnées du contour pour le texte
        x, y, w, h = cv2.boundingRect(cnt)

        # Dessiner le contour et écrire le nom de la forme
        cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Extraire la région d'intérêt (ROI) de l'image propre
        roi = image_clean[y:y+h, x:x+w]

        # Vérifier si la ROI est valide
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"ROI vide à {x},{y} — ignoré")
            continue

        # Ajouter une condition pour ne pas sauvegarder les formes inconnues

        # Sauvegarder la ROI dans le dossier de sortie
        filename = os.path.join(output_dir, f"{shape}_{x}_{y}.jpg")
        cv2.imwrite(filename, roi)
        print(f"Sauvegarde : {filename}")

    return image