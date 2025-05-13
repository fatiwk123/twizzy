import cv2
import numpy as np
import os

def preprocess_image(image, target_size=(100, 100)):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Redimensionner
    resized = cv2.resize(gray, target_size)
    return resized

def template_matching_orb(extraits_dir = 'panneaux_extraits', templates_dir = 'templates', target_size=(100, 100)):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matched_list = []
    # === 1. Charger les templates avec noms dans un dictionnaire ===
    templates = {}
    for file in os.listdir(templates_dir):
        if file.endswith((".jpg", ".png")):
            template_path = os.path.join(templates_dir, file)
            template_image = cv2.imread(template_path)
            if template_image is not None:
                processed = preprocess_image(template_image, target_size)
                templates[file] = processed

    # === 2. Parcourir les extraits à comparer ===
    for fichier_extrait in os.listdir(extraits_dir):
        if fichier_extrait.endswith((".jpg", ".png")):
            extrait_path = os.path.join(extraits_dir, fichier_extrait)
            extrait_image = cv2.imread(extrait_path)
            if extrait_image is None:
                continue

            extrait_processed = preprocess_image(extrait_image, target_size)
            kp_extrait, des_extrait = orb.detectAndCompute(extrait_processed, None)

            if des_extrait is None:
                print(f"[AVERTISSEMENT] Pas de descripteurs pour {fichier_extrait}")
                continue

            # Variables pour retenir le meilleur match
            best_match_count = 0
            best_template_name = None
            best_template = None
            best_matches = None
            best_kp_template = None

            # === 3. Comparer avec chaque template ===
            for name, template_processed in templates.items():
                kp_template, des_template = orb.detectAndCompute(template_processed, None)
                if des_template is None:
                    continue

                matches = bf.match(des_extrait, des_template)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > best_match_count:
                    best_match_count = len(matches)
                    best_template_name = name
                    best_template = template_processed
                    best_matches = matches
                    best_kp_template = kp_template

            # === 4. Vérifier si le match est suffisant ===
            SEUIL_BONNES_CORRESPONDANCES = 10
            if best_template is not None and best_match_count >= SEUIL_BONNES_CORRESPONDANCES:
                print(f"[MATCH] {fichier_extrait} correspond à {best_template_name} avec {best_match_count} correspondances.")

                # Dessiner les correspondances
                result = cv2.drawMatches(
                    extrait_processed, kp_extrait,
                    best_template, best_kp_template,
                    best_matches[:20], None, flags=2
                )

                # Ajouter le nom du panneau reconnu en haut de l'image
                cv2.putText(result, f"Match: {best_template_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Afficher l'image avec le nom du match
                matched_list.append(result)

    return matched_list
