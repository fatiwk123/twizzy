import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


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

        # Bouton pour lancer traitement
        self.btn_treatment = tk.Button(
            button_frame,
            text="Launch treatment",
            command=self.image_treatment_placeholder,
        )
        self.btn_treatment.pack(side=tk.LEFT, padx=5)

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

    def image_treatment_placeholder(self):
        """Process current image"""
        return 0
