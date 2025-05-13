import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from image_viewer import ImageViewer
from video_viewer import VideoViewer


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Twizzy Main Window")

        # Widget pour contenir boutons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Bouton pour choisir ImageViewer
        self.btn_image = tk.Button(
            button_frame, text="Image", command=self.create_image_viewer
        )
        self.btn_image.pack(side=tk.LEFT, padx=5)

        # Bouton pour choisir VideoViewer
        self.btn_video = tk.Button(
            button_frame,
            text="Video",
            command=self.create_video_viewer,
        )
        self.btn_video.pack(side=tk.LEFT, padx=5)
        button_frame.pack_configure(anchor=tk.CENTER)

        self.image_viewer_opened = None
        self.video_viewer_opened = None

    def create_image_viewer(self):
        """creates and displays image viewer"""
        new_window = tk.Toplevel(root)
        image_viewer = ImageViewer(new_window)
        self.image_viewer_opened = image_viewer
        return image_viewer

    def create_video_viewer(self):
        """creates and displays video viewer"""
        new_window = tk.Toplevel(root)
        video_viewer = VideoViewer(new_window)
        self.video_viewer_opened = video_viewer
        return video_viewer


# Run the app; Put in a main file later
root = tk.Tk()
app = MainWindow(root)
root.mainloop()
