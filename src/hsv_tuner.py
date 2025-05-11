import cv2
import numpy as np

# Charger l'image
image = cv2.imread('images/image_test__final/GOPR0156.jpg')
if image is None:
    print("Erreur chargement image.")
    exit()

# Redimensionner (optionnel)
image = cv2.resize(image, (640, 480))

# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Convertir en HSV
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Fenêtre interactive
cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)

def nothing(x): pass

# Curseurs HSV
cv2.createTrackbar("H min", "Tuner", 0, 180, nothing)
cv2.createTrackbar("H max", "Tuner", 180, 180, nothing)
cv2.createTrackbar("S min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("S max", "Tuner", 255, 255, nothing)
cv2.createTrackbar("V min", "Tuner", 0, 255, nothing)
cv2.createTrackbar("V max", "Tuner", 255, 255, nothing)

while True:
    # Lire curseurs
    h_min = cv2.getTrackbarPos("H min", "Tuner")
    h_max = cv2.getTrackbarPos("H max", "Tuner")
    s_min = cv2.getTrackbarPos("S min", "Tuner")
    s_max = cv2.getTrackbarPos("S max", "Tuner")
    v_min = cv2.getTrackbarPos("V min", "Tuner")
    v_max = cv2.getTrackbarPos("V max", "Tuner")

    # Masque HSV
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphologie (nettoyage masque)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # supprime le bruit
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # ferme les trous

    # Affichage
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("Masque", mask)
    cv2.imshow("Résultat", result)

    key = cv2.waitKey(1)
    if key == 27:
        print(f"\n✅ HSV sélectionné :")
        print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")
        break

cv2.destroyAllWindows()
