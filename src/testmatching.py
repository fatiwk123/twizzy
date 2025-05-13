import cv2
import numpy as np
import os

# === Prétraitement d'une image ===
def preprocess_image(image, target_size=(100, 100)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    return resized

# === Fonction principale de reconnaissance ORB ===
def template_matching_orb(extraits_dir, templates_dir, target_size=(100, 100)):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # === 1. Chargement des templates ===
    templates = {}
    for file in os.listdir(templates_dir):
        if file.endswith((".jpg", ".png")):
            template_path = os.path.join(templates_dir, file)
            template_image = cv2.imread(template_path)
            if template_image is not None:
                processed = preprocess_image(template_image, target_size)
                kp_tpl, des_tpl = orb.detectAndCompute(processed, None)
                if des_tpl is not None:
                    templates[file] = (processed, kp_tpl, des_tpl)

    # === 2. Parcours des extraits à comparer ===
    for fichier_extrait in os.listdir(extraits_dir):
        if fichier_extrait.endswith((".jpg", ".png")):
            extrait_path = os.path.join(extraits_dir, fichier_extrait)
            extrait_image = cv2.imread(extrait_path)
            if extrait_image is None:
                continue

            extrait_processed = preprocess_image(extrait_image, target_size)
            kp_extrait, des_extrait = orb.detectAndCompute(extrait_processed, None)

            if des_extrait is None:
                print(f"[!] Pas de descripteurs pour {fichier_extrait}")
                continue

            # === 3. Matching avec tous les templates ===
            best_match_name = None
            best_match_count = 0
            best_result = None
            best_kp_template = None
            best_matches = []

            for name, (tpl_img, kp_tpl, des_tpl) in templates.items():
                matches = bf.match(des_extrait, des_tpl)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > best_match_count:
                    best_match_count = len(matches)
                    best_match_name = name
                    best_result = tpl_img
                    best_kp_template = kp_tpl
                    best_matches = matches

            # === 4. Affichage du meilleur match si suffisant ===
            SEUIL_MATCH = 10
            if best_match_name and best_match_count >= SEUIL_MATCH:
                print(f"[MATCH] {fichier_extrait} → {best_match_name} ({best_match_count} correspondances)")

                result_img = cv2.drawMatches(
                    extrait_processed, kp_extrait,
                    best_result, best_kp_template,
                    best_matches[:20], None, flags=2
                )
                cv2.putText(result_img, f"Match: {best_match_name}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(f"Matching - {fichier_extrait}", result_img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

# === Paramètres à adapter ===
extraits_dir = "panneaux_extraits"
templates_dir = "templates"

# === Lancer la reconnaissance ===
if __name__ == "__main__":
    template_matching_orb(extraits_dir, templates_dir)
