import cv2
import numpy as np
import os

def preprocess_image(image, target_size=(100, 100)):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Redimensionner l'image
    resized = cv2.resize(gray, target_size)

    return resized

def template_matching_orb(extraits_dir, templates_dir, target_size=(100, 100)):
    # Initialiser le détecteur ORB et le matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Charger les templates et les pré-traiter
    templates = []
    for file in os.listdir(templates_dir):
        if file.endswith((".jpg", ".png")):
            template_path = os.path.join(templates_dir, file)
            template = cv2.imread(template_path)
            if template is not None:
                template = preprocess_image(template, target_size)
                templates.append(template)

    # Parcourir les extraits pour comparaison
    for fichier_extrait in os.listdir(extraits_dir):
        if fichier_extrait.endswith((".jpg", ".png")):
            # Charger et pré-traiter l'extrait
            extrait_path = os.path.join(extraits_dir, fichier_extrait)
            extrait = cv2.imread(extrait_path)
            if extrait is None:
                continue
            extrait = preprocess_image(extrait, target_size)

            # Détecter les points clés et descripteurs de l'extrait
            kp_extrait, des_extrait = orb.detectAndCompute(extrait, None)

            # Initialiser les variables pour le meilleur match
            best_match_count = 0
            best_template = None
            best_matches = None

            # Parcourir les templates
            for template in templates:
                # Détecter les points clés et descripteurs du template
                kp_template, des_template = orb.detectAndCompute(template, None)

                # Passer si aucun descripteur n'est trouvé
                if des_template is None:
                    continue

                # Trouver les correspondances
                matches = bf.match(des_extrait, des_template)

                # Trier les correspondances par distance
                matches = sorted(matches, key=lambda x: x.distance)

                # Mettre à jour le meilleur match
                if len(matches) > best_match_count:
                    best_match_count = len(matches)
                    best_template = template
                    best_matches = matches
                    best_kp_template = kp_template

            # Seuil de nombre de bonnes correspondances
            SEUIL_BONNES_CORRESPONDANCES = 10  # Ajuster ce seuil ici

            # Afficher les résultats
            if best_template is not None and best_match_count > SEUIL_BONNES_CORRESPONDANCES:
                # Dessiner les correspondances
                result = cv2.drawMatches(extrait, kp_extrait, best_template, best_kp_template, best_matches[:20], None, flags=2)

                # Afficher le résultat
                cv2.imshow("Meilleur Matching", result)
                cv2.waitKey(0)
            else:
                print(f"Aucun match fiable pour {fichier_extrait}")

    cv2.destroyAllWindows()

# Chemin des dossiers contenant les extraits et les templates
extraits_dir = 'panneaux_extraits'
templates_dir = 'templates'

# Appeler la fonction
template_matching_orb(extraits_dir, templates_dir)