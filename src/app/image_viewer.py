import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        
        # Bouton pour lancer affichage
        self.btn_select = tk.Button(root, text="Select Image", command=self.select_image)
        self.btn_select.pack(pady=10)
        
        # Zone pour afficher image
        self.canvas = tk.Canvas(root, width=800, height=800, bg="white")
        self.canvas.pack()
        
        # Placeholder pour enregistrer l'image
        self.image_tk = None

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.display_image(file_path) #On lance la méthode display_image si on a bien selectionne une image

    def display_image(self, file_path):
        # Ouvre et redimensionne l'image pour qu'elle tienne dans le canvas (optionnel)
        img_pil = Image.open(file_path)
        img_pil = img_pil.resize((800, 800), Image.LANCZOS)  # redimensionnement (peut ne pas etre adapte pour images rectangles)
        
        # Conversion en format tkinter
        self.image_tk = ImageTk.PhotoImage(img_pil)
        
        # Affichage
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

# Run the app
root = tk.Tk()
app = ImageViewer(root)
root.mainloop()