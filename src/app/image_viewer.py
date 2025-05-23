import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys
import os
import cv2

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import detection
import hsv_tuner
import matching

# todo : create docstring for this class


# classe qui permet de contenir et afficher une image tkinter obtenue a partir de l'explo windows
class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")

        # Widget pour contenir boutons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Bouton pour lancer affichage
        self.btn_select = tk.Button(
            button_frame, text="Select New Image", command=self.select_image
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)

        # Bouton pour lancer traitement contours
        self.btn_treatment_contours = tk.Button(
            button_frame,
            text="Launch Contour Detection",
            command=self.image_treatment_contours,
        )
        self.btn_treatment_contours.pack(side=tk.LEFT, padx=5)

        # Bouton pour lancer traitement hsv
        self.btn_treatment_hsv = tk.Button(
            button_frame,
            text="Launch HSV Treatment",
            command=self.image_treatment_hsv,
        )
        self.btn_treatment_hsv.pack(side=tk.LEFT, padx=5)

        # Bouton pour lancer tmatching
        self.btn_treatment_matching = tk.Button(
            button_frame,
            text="Launch Matching",
            command=self.image_treatment_matching,
        )
        self.btn_treatment_matching.pack(side=tk.LEFT, padx=5)

        button_frame.pack_configure(anchor=tk.CENTER)

        # Zone pour afficher image
        self.canvas = tk.Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack()

        # Placeholder pour enregistrer l'image et son path
        self.image_tk = None
        self.image_opened_path = None

    def select_image(self):
        """Selects an image using file explorer"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if file_path:
            self.display_image(
                file_path
            )  # On lance la méthode display_image si on a bien selectionne une image

    def display_image(self, file_path):
        """Displays the selected image in the frame"""
        img_pil = Image.open(file_path)
        img_pil = img_pil.resize(
            (800, 800), Image.LANCZOS
        )  # redimensionnement (peut ne pas etre adapte pour images rectangles)

        # Conversion en format tkinter
        self.image_tk = ImageTk.PhotoImage(img_pil)
        self.image_opened_path = file_path
        # Affichage
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

    def image_treatment_contours(self):
        """Process contours current image"""

        if self.image_tk == None:
            print("Erreur : Aucune image chargée")
        else:
            # cv2 image
            image_contours = detection.detect_and_extract_shapes(self.image_opened_path)
            height, width = image_contours.shape[:2]
            new_window = tk.Toplevel(self.root)
            new_window.title("Contours")
            canvas = tk.Canvas(new_window, width=width, height=height, bg="white")
            canvas.pack()

            image_contours_rgb = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)
            image_contours_pil = Image.fromarray(image_contours_rgb)
            image_contours_tkinter = ImageTk.PhotoImage(image=image_contours_pil)
            canvas.create_image(0, 0, anchor="nw", image=image_contours_tkinter)
            canvas.image = image_contours_tkinter

    def image_treatment_hsv(self):
        """Process hsv current image"""

        if self.image_tk == None:
            print("Erreur : Aucune image chargée")
        else:
            # cv2 image
            image_hsv = hsv_tuner.hsv_tuner(self.image_opened_path)
            height, width = image_hsv.shape[:2]
            new_window = tk.Toplevel(self.root)
            new_window.title("HSV")
            canvas = tk.Canvas(new_window, width=width, height=height, bg="white")
            canvas.pack()

            image_hsv_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2RGB)
            image_hsv_pil = Image.fromarray(image_hsv_rgb)
            image_hsv_tkinter = ImageTk.PhotoImage(image=image_hsv_pil)
            canvas.create_image(0, 0, anchor="nw", image=image_hsv_tkinter)
            canvas.image = image_hsv_tkinter

    def image_treatment_matching(self):
        if self.image_tk == None:
            print("Erreur : Aucune image chargée")
        else:
            matched_list = matching.template_matching_orb()
            for matched in matched_list:

                height, width = matched.shape[:2]
                new_window = tk.Toplevel(self.root)
                new_window.title("Match")
                canvas = tk.Canvas(new_window, width=width, height=height, bg="white")
                canvas.pack()

                image_match_rgb = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
                image_match_pil = Image.fromarray(image_match_rgb)
                image_match_tkinter = ImageTk.PhotoImage(image=image_match_pil)
                canvas.create_image(0, 0, anchor="nw", image=image_match_tkinter)
                canvas.image = image_match_tkinter
