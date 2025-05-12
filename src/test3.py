import cv2
import numpy as np


# Créer une matrice identité 3x3 de type uint8
mat = np.eye(3, dtype=np.uint8)


print("mat =")
print(mat)




#def lire_image(fichier):
    #return cv2.imread(fichier)


# Exemple d'utilisation
#fichier_image = "images/image_open_CV/circles.jpg"
#mat = lire_image(fichier_image)
#print(mat)




def lire_image(fichier):
    return cv2.imread(fichier)


#def main():
 #   mat = lire_image("images/image_open_CV/opencv.png")
  #  if mat is not None:
   #     for i in range(mat.shape[0]):
    #        for j in range(mat.shape[1]):
     #           BGR = mat[i, j]
      #          if BGR[0] == 255 and BGR[1] == 255 and BGR[2] == 255:
       #             print(".", end="")
        #        else:
         #           print("+", end="")
          #  print()
import cv2


def main():
    img = cv2.imread("images/image_open_CV/bgr.png")
    if img is not None:
        b, g, r = cv2.split(img)
       
        # Afficher les canaux dans des fenêtres séparées
        cv2.imshow("Bleu", b)
        cv2.imshow("Vert", g)
        cv2.imshow("Rouge", r)
       
        # Attendre une touche et fermer toutes les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")






def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/bgr.png")
   
    if img is not None:
        # Séparer les canaux B, G, R
        b, g, r = cv2.split(img)
       
        # Créer une matrice vide de mêmes dimensions que l'image
        empty = np.zeros_like(b)
       
        # Afficher chaque canal individuellement
        cv2.imshow("Bleu", b)
        cv2.imshow("Vert", g)
        cv2.imshow("Rouge", r)
       
        # Créer et afficher les combinaisons de canaux
        cv2.imshow("Bleu seul", cv2.merge([b, empty, empty]))
        cv2.imshow("Vert seul", cv2.merge([empty, g, empty]))
        cv2.imshow("Rouge seul", cv2.merge([empty, empty, r]))
       
        # Attendre une touche et fermer toutes les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")
import cv2
import numpy as np


def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/hsv.png")
   
    if img is not None:
        # Convertir l'image de BGR à HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
        # Afficher l'image convertie
        cv2.imshow("HSV", hsv)
       
        # Séparer les canaux HSV
        h, s, v = cv2.split(hsv)
       
        # Afficher les canaux individuels
        cv2.imshow("Hue", h)
        cv2.imshow("Saturation", s)
        cv2.imshow("Value", v)
       
        # Définir les valeurs HSV à tester
        hsv_values = np.array([[1, 255, 255], [179, 1, 255], [179, 0, 1]])
       
        for i in range(3):
            # Afficher chaque canal
            cv2.imshow(f"{i}-HSV", hsv[:, :, i])
           
            # Créer une matrice pour les canaux
            channels = [np.ones_like(h), np.ones_like(s), np.ones_like(v)]
           
            for j in range(3):
                # Créer une matrice avec la valeur HSV spécifiée
                channels[j] = np.full_like(hsv[:, :, j], hsv_values[i][j])
           
            # Fusionner les canaux
            dst = cv2.merge(channels)
           
            # Convertir de HSV à BGR pour affichage
            res = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
           
            # Afficher le résultat
            cv2.imshow(f"Result-{i}", res)
       
        # Attendre une touche et fermer toutes les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")


import cv2
import numpy as np


def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/circles.jpg")
   
    if img is not None:
        # Convertir l'image de BGR à HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
        # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
       
        # Créer une image masque où les pixels dans la plage de rouge sont blancs (255), sinon noirs (0)
        threshold_img = cv2.inRange(hsv_image, lower_red, upper_red)
       
        # Appliquer un flou gaussien pour réduire le bruit
        threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
       
        # Afficher l'image traitée
        cv2.imshow("Cercles rouges", threshold_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")


import cv2
import numpy as np


def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/circles.jpg")
   
    if img is not None:
        # Convertir l'image de BGR à HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
       
        # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
        # Rouge basse teinte
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
       
        # Rouge haute teinte
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
       
        # Créer des images masque pour les deux plages de rouge
        threshold_img1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        threshold_img2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
       
        # Combiner les deux images masque
        threshold_img = cv2.bitwise_or(threshold_img1, threshold_img2)
       
        # Appliquer un flou gaussien pour réduire le bruit
        threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
       
        # Afficher l'image traitée
        cv2.imshow("Cercles rouges", threshold_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")








import cv2
import numpy as np
import random


def detecter_cercles(img):
    # Convertir l'image de BGR à HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
   
    # Créer des images masque pour les deux plages de rouge
    threshold_img1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    threshold_img2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
   
    # Combiner les deux images masque
    threshold_img = cv2.bitwise_or(threshold_img1, threshold_img2)
   
    # Appliquer un flou gaussien pour réduire le bruit
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
   
    return threshold_img


def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/circles.jpg")
   
    if img is not None:
        # Afficher l'image originale
        cv2.imshow("Cercles", img)
       
        # Détecter les cercles
        threshold_img = detecter_cercles(img)
       
        # Afficher l'image après seuillage
        cv2.imshow("Seuillage", threshold_img)
       
        # Appliquer le détecteur de contours Canny
        canny_output = cv2.Canny(threshold_img, 100, 200)
       
        # Trouver les contours
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Créer une image pour dessiner les contours
        drawing = np.zeros_like(img)
       
        # Dessiner les contours
        for i, contour in enumerate(contours):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(drawing, contours, i, color, 1)
       
        # Afficher les contours dessinés
        cv2.imshow("Contours", drawing)
       
        # Attendre une touche et fermer toutes les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")
import cv2
import numpy as np


def detecter_cercles(img):
    # Convertir l'image de BGR à HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
   
    # Créer des images masque pour les deux plages de rouge
    threshold_img1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    threshold_img2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
   
    # Combiner les deux images masque
    threshold_img = cv2.bitwise_or(threshold_img1, threshold_img2)
   
    # Appliquer un flou gaussien pour réduire le bruit
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
   
    return threshold_img


def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/circles_rectangles.jpg")
   
    if img is not None:
        # Afficher l'image originale
        cv2.imshow("Cercles", img)
       
        # Détecter les cercles
        threshold_img = detecter_cercles(img)
       
        # Afficher l'image après seuillage
        cv2.imshow("Seuillage", threshold_img)
       
        # Trouver les contours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Dessiner les cercles détectés
        for contour in contours:
            # Calculer l'aire du contour
            contour_area = cv2.contourArea(contour)
           
            # Trouver le cercle minimum englobant
            (center, radius) = cv2.minEnclosingCircle(contour)
           
            # Calculer le rapport aire/cercle pour vérifier si c'est un cercle
            circle_area = np.pi * radius * radius
            if circle_area > 0 and contour_area / circle_area >= 0.8:
                # Dessiner le cercle
                cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
       
        # Afficher l'image avec les cercles détectés
        cv2.imshow("Détection des cercles rouges", img)
       
        # Attendre une touche et fermer toutes les fenêtres
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")


import cv2
import numpy as np

def detecter_cercles(img):
    # Convertir l'image de BGR à HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    # Créer des images masque pour les deux plages de rouge
    threshold_img1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    threshold_img2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # Combiner les deux images masque
    threshold_img = cv2.bitwise_or(threshold_img1, threshold_img2)
    
    # Appliquer un flou gaussien pour réduire le bruit
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
    
    return threshold_img

def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/Billard_Balls.jpg")
    
    if img is not None:
        # Détecter les balles rouges
        threshold_img = detecter_cercles(img)
        
        # Trouver les contours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dessiner les contours sur une image noire
        drawing = np.zeros_like(img)
        for i, contour in enumerate(contours):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.drawContours(drawing, contours, i, color, 1)
        
        # Dessiner les cercles et rectangles détectés
        for contour in contours:
            # Calculer l'aire du contour
            contour_area = cv2.contourArea(contour)
            
            # Ignorer les petits contours
            if contour_area < 1000:  # Ajustez cette valeur en fonction de votre image
                continue
            
            # Ignorer les contours avec une forme irrégulière
            (center, radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            if circle_area > 0 and contour_area / circle_area < 0.8:
                continue
            
            # Dessiner le cercle et le rectangle
            cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
            rect = cv2.boundingRect(contour)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
            
            # Extraire la ROI
            roi = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            ball = np.zeros_like(roi)
            ball[:] = roi[:]
            cv2.imshow("Balle", ball)
        
        # Afficher les résultats
        cv2.imshow("Détection des balles", img)
        cv2.imshow("Contours", drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")

if __name__ == "__main__":
    main()
import cv2
import numpy as np

def main():
    # Charger l'image de la signalisation routière et l'objet à détecter
    sroadSign = cv2.imread("images/image_open_CV/Billard_Balls.jpg")
    objectfile = cv2.imread("images/image_open_CV/Ball_13.png")
    
    if sroadSign is not None and objectfile is not None:
        # Redimensionner l'objet pour qu'il ait la même taille que la signalisation routière
        sObject = cv2.resize(objectfile, (sroadSign.shape[1], sroadSign.shape[0]))
        
        # Convertir les images en échelle de gris
        grayObject = cv2.cvtColor(sObject, cv2.COLOR_BGR2GRAY)
        graySign = cv2.cvtColor(sroadSign, cv2.COLOR_BGR2GRAY)
        
        # Normaliser les images
        grayObject = cv2.normalize(grayObject, None, 0, 255, cv2.NORM_MINMAX)
        graySign = cv2.normalize(graySign, None, 0, 255, cv2.NORM_MINMAX)
        
        # Initialiser le détecteur et l'extracteur de caractéristiques ORB
        orbDetector = cv2.ORB_create()
        
        # Détecter les points clés et calculer les descripteurs pour l'objet
        objectKeypoints, objectDescriptors = orbDetector.detectAndCompute(grayObject, None)
        
        # Détecter les points clés et calculer les descripteurs pour la signalisation routière
        signKeypoints, signDescriptors = orbDetector.detectAndCompute(graySign, None)
        
        # Initialiser le matcher de descripteurs
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Effectuer le matching entre les descripteurs de l'objet et ceux de la signalisation
        matches = matcher.match(objectDescriptors, signDescriptors)
        
        # Trier les appariements par distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Afficher les appariements
        print("Matches:", len(matches))
        
        # Dessiner les appariements sur une image
        matchedImage = cv2.drawMatches(sObject, objectKeypoints, sroadSign, signKeypoints, matches[:10], None, flags=2)
        
        # Afficher l'image des appariements
        cv2.imshow("Matched Features", matchedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Une des images n'a pas été trouvée ou impossible à charger.")

if __name__ == "__main__":
    main()
import cv2
import numpy as np

def detecter_balles(img):
    # Convertir l'image de BGR à HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Définir les bornes inférieure et supérieure pour la couleur rouge dans l'espace HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    # Créer des images masque pour les deux plages de rouge
    threshold_img1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    threshold_img2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    
    # Combiner les deux images masque
    threshold_img = cv2.bitwise_or(threshold_img1, threshold_img2)
    
    # Appliquer un flou gaussien pour réduire le bruit
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), 2)
    
    return threshold_img

def main():
    # Charger l'image
    img = cv2.imread("images/image_open_CV/Billard_Balls.jpg")
    
    if img is not None:
        # Détecter les balles
        threshold_img = detecter_balles(img)
        
        # Trouver les contours
        contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dessiner les contours sur une image noire
        drawing = np.zeros_like(img)
        for i, contour in enumerate(contours):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.drawContours(drawing, contours, i, color, 1)
        
        # Dessiner les cercles et rectangles détectés et identifier les balles
        for contour in contours:
            # Calculer l'aire du contour
            contour_area = cv2.contourArea(contour)
            
            # Ignorer les petits contours
            if contour_area < 1000:  # Ajustez cette valeur en fonction de votre image
                continue
            
            # Ignorer les contours avec une forme irrégulière
            (center, radius) = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            if circle_area > 0 and contour_area / circle_area < 0.8:
                continue
            
            # Dessiner le cercle et le rectangle
            cv2.circle(img, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
            rect = cv2.boundingRect(contour)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
            
            # Extraire la ROI
            roi = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            
            # Extraire la balle 13
            ball = np.zeros_like(roi)
            ball[:] = roi[:]
            cv2.imshow("Balle", ball)
        
        # Afficher les résultats
        cv2.imshow("Détection des balles", img)
        cv2.imshow("Contours", drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image non trouvée ou impossible à charger.")

if __name__ == "__main__":
    main()