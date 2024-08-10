import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
model=keras.models.load_model('doodle-10-2.h5')
class DoodlePad:
    def __init__(self, root, width=280, height=280):
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        
        self.image = Image.new("L", (self.width, self.height), color=255)
        self.draw = ImageDraw.Draw(self.image)

       
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=10)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def get_bitmap(self):
        
        resized_image = self.image.resize((28, 28))
        bitmap = np.array(resized_image)

       
        bitmap = 255 - bitmap

        return bitmap

    def save_bitmap_as_npy(self, file_path):
        
        bitmap = self.get_bitmap()
        np.save(file_path, bitmap)
        print(f"Bitmap saved as {file_path}.npy")
def detect_shape(file):
     class_names=['circle','triangle','moon', 'line', 'smiley_face', 'hexagon', 'square', 'octagon', 'umbrella', 'star'] 
     data=np.load(file)
     data=data.reshape(28,28,1).astype('float32')
     pred = model.predict(np.expand_dims(data, axis=0))[0]
     ind = (-pred).argsort()[:1]
     latex = [class_names[x] for x in ind]
     print(latex[0])



if __name__ == "__main__":
    root = tk.Tk()
    doodle_pad = DoodlePad(root)

    def save_bitmap_as_npy():
        file_path = "doodle_bitmap"  
        doodle_pad.save_bitmap_as_npy(file_path)

    button_save_npy = tk.Button(root, text="Save as .npy", command=save_bitmap_as_npy)
    button_save_npy.pack()

    root.mainloop()
    detect_shape("doodle_bitmap.npy")
