import cv2
import os
import numpy as np

# Dossiers
path_extraits = "twizzy/panneaux_extraits"
path_templates = "twizzy/templates"

# Initialiser ORB
orb = cv2.ORB_create()

# Charger les modèles
model_descriptors = {}
for filename in os.listdir(path_templates):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(path_templates, filename)
        img = cv2.imread(img_path, 0)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            model_descriptors[filename] = (img, kp, des)

# Pour chaque panneau extrait
for extrait_file in os.listdir(path_extraits):
    if not extrait_file.lower().endswith((".jpg", ".png")):
        continue

    extrait_path = os.path.join(path_extraits, extrait_file)
    img_extrait = cv2.imread(extrait_path, 0)
    kp_extrait, des_extrait = orb.detectAndCompute(img_extrait, None)

    if des_extrait is None:
        print(f"[!] Aucun descripteur trouvé pour {extrait_file}")
        continue

    # Comparaison avec tous les modèles
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_score = 0
    best_match = None
    best_img = None
    best_kp = None
    best_matches = []

    for modele_name, (img_model, kp_model, des_model) in model_descriptors.items():
        matches = bf.match(des_extrait, des_model)
        score = len(matches)
        if score > best_score:
            best_score = score
            best_match = modele_name
            best_img = img_model
            best_kp = kp_model
            best_matches = matches

    # Affichage du résultat
    print(f"{extrait_file} → Reconnu comme : {best_match} ({best_score} correspondances)")

    # Affichage visuel
    if best_match:
        matched_visu = cv2.drawMatches(img_extrait, kp_extrait, best_img, best_kp, best_matches[:20], None, flags=2)
        cv2.imshow(f"{extrait_file} → {best_match}", matched_visu)
        cv2.waitKey(0)

cv2.destroyAllWindows()
