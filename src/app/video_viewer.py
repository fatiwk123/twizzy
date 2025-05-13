import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import matching_video

class VideoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Viewer")

        # Video variables
        self.video_path = None
        self.cap = None  # OpenCV video capture object
        self.is_playing = False
        self.current_frame = 0

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        # Control buttons (horizontal layout)
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Frame for video display
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(pady=10)

        self.btn_open = tk.Button(
            button_frame, text="Open Video", command=self.open_video
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(
            button_frame, text="Play/Pause", command=self.toggle_play
        )
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_matching = tk.Button(
            button_frame, text="Match", command=self.process_video
        )
        self.btn_matching.pack(side=tk.LEFT, padx=5)

    def open_video(self):
        """Load a video file using OpenCV."""
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.show_frame(0)  # Show first frame

    def show_frame(self, frame_num):
        """Display a specific frame in the Tkinter window."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                # Convert OpenCV BGR to RGB and resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((800, 600))  # Resize for display
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_frame.config(image=imgtk)
                self.video_frame.image = imgtk  # Keep reference
                self.current_frame = frame_num

    def toggle_play(self):
        """Play/pause video by updating frames periodically."""
        if not self.cap:
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_video()

    def play_video(self):
        """Update frames to simulate playback."""
        if self.is_playing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.show_frame(self.current_frame + 1)
                self.root.after(30, self.play_video)  # ~30 FPS
            else:
                self.is_playing = False

    def process_video(self):
        """Opens a cv2 video with active matching"""
        
        if self.video_path == None:
            print('Erreur : Aucune video chargée')
        elif self.is_playing != False:
            print('Erreur : Mettez la video en pause avant de lancer le matching')
        else:
            matching_video.match_video(self.video_path)
        
