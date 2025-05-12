import cv2
import numpy as np

def load_image(image_path):
    """Charge une image et la convertit en niveaux de gris."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"L'image n'a pas été trouvée ou le chemin est incorrect : {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def apply_filters(gray_image):
    """Applique des filtrages et débruitage."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred

def segment_image(blurred_image):
    """Segmentation et seuillage (couleurs)."""
    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_contours(thresh_image):
    """Détection de contours (formes géométriques)."""
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_features(contours):
    """Extraction de caractéristiques pour la reconnaissance de formes."""
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        features.append((area, aspect_ratio, x, y, w, h))
    return features

def template_matching(image, template):
    """Introduction aux algorithmes de template Matching."""
    if template is None or template.size == 0:
        return None

    h, w = template.shape[:2]
    if h > image.shape[0] or w > image.shape[1]:
        return None

    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.8:
        return max_loc
    return None

def detect_shapes(contours):
    """Détection d'une forme géométrique : cercle, rectangle, carré, polygone, triangle."""
    shapes = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shapes.append("Triangle")
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if aspect_ratio > 0.95 and aspect_ratio < 1.05:
                shapes.append("Rectangle")
            else:
                shapes.append("Square")
        elif len(approx) > 4:
            shapes.append("Polygon")
        else:
            shapes.append("Unknown")
    return shapes

def extract_object(image, contour):
    """Extraction d'un objet depuis une image."""
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

def resize_image(image, scale=0.5):
    """Réduit la taille de l'image."""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width, height)
    resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return resized

def main():
    image_path = 'images/image_test__final/G0030141.jpg'
    template_path = 'templates/template.jpg'  # Assurez-vous que ce chemin est correct

    image, gray = load_image(image_path)
    blurred = apply_filters(gray)
    thresh = segment_image(blurred)
    contours = detect_contours(thresh)
    features = extract_features(contours)
    template = cv2.imread(template_path, 0)
    template_loc = template_matching(gray, template)

    shapes = detect_shapes(contours)
    for i, contour in enumerate(contours):
        object_image = extract_object(image, contour)
        resized_object = resize_image(object_image, scale=0.5)
        cv2.imshow(f"Object {i}", resized_object)
        print(f"Shape: {shapes[i] if i < len(shapes) else 'Unknown'}")

    if template_loc and template is not None:
        x, y = template_loc
        w, h = template.shape[1], template.shape[0]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    resized_image = resize_image(image, scale=0.5)
    resized_thresh = resize_image(thresh, scale=0.5)
    cv2.imshow("Thresholded Image", resized_thresh)
    cv2.imshow("Detected Shapes", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()