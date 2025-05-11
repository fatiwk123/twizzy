import cv2
import numpy as np
import os

def detect_panneaux(image_path):
    os.makedirs("panneaux_extraits", exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print("Erreur : image non chargée")
        return

    image = cv2.resize(image, (640, 480))
    original = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    lower_red1 = np.array([0, 50, 20])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([160, 50, 20])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

        if circularity > 0.85:
            shape = "Cercle"
        elif len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            ratio = w / float(h)
            shape = "Carre" if 0.95 <= ratio <= 1.05 else "Rectangle"
        elif 7 <= len(approx) <= 9:
            shape = "Octogone"
        else:
            shape = "Forme"

        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        roi = original[y:y+h, x:x+w]
        filename = f"panneaux_extraits/{shape}_{x}_{y}.png"
        cv2.imwrite(filename, roi)
        count += 1

    # Détection de cercles Hough (améliorée)
    gray_blurred = cv2.medianBlur(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 5)
    masked_gray = cv2.bitwise_and(gray_blurred, gray_blurred, mask=red_mask)

    circles = cv2.HoughCircles(
        masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
        param1=100, param2=35, minRadius=20, maxRadius=80
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i[0], i[1], i[2]
            if 0 <= cy - r < cy + r <= image.shape[0] and 0 <= cx - r < cx + r <= image.shape[1]:
                cv2.circle(image, (cx, cy), r, (0, 255, 255), 2)
                roi = original[cy - r:cy + r, cx - r:cx + r]
                filename = f"panneaux_extraits/Cercle_Hough_{cx}_{cy}.png"
                cv2.imwrite(filename, roi)

    cv2.imshow("Image originale", original)
    cv2.imshow("Masque rouge", red_mask)
    cv2.imshow("Formes detectees", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"{count} panneaux extraits et sauvegardés.")

# Lancer
detect_panneaux("images/image_test__final/GOPR0156.jpg")
