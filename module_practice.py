import tkinter as tk
from tkinter import filedialog
import pydicom as dicom
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

class DICOMViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM VIEWER: BME213")
        
        self.image = None
        self.original_image = None
        
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()
        
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()
        
        tk.Button(self.button_frame, text="Open File", width=10, height=5, command=self.load_dicom).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Original", width=10, height=5, command=self.show_original).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Binary", width=10, height=5, command=self.apply_otsu).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Average filter", width=10, height=5, command=self.avg_filter).pack(side=tk.LEFT)
        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

    def load_dicom(self):
        file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
        if file_path:
            ds = dicom.dcmread(file_path)
            original = ds.pixel_array
            original = original.astype(float)
            original = original / np.max(original)
            original = original * 255
            
            self.original_image = original
            self.image = original
            
            self.ax.clear()
            self.ax.imshow(self.image, cmap='gray')
            self.canvas.draw()

    def show_original(self):
        target_path = r"C:\Users\yoonc\OneDrive\바탕 화면\3학년\기초프로그래밍\1-11.dcm"
        ds = dicom.dcmread(target_path)
        original = ds.pixel_array
        original = original.astype(float)
        original = original / np.max(original)
        original = original * 255
        
        self.original_image = original
        self.image = original
        
        self.ax.clear()
        self.ax.imshow(self.original_image, cmap='gray')
        self.canvas.draw()

    def compute_intra_class_variance(self, image, threshold):
        class1 = image[image < threshold]
        class2 = image[image >= threshold]
        
        if len(class1) == 0 or len(class2) == 0:
            return float('inf')
            
        w1 = len(class1) / image.size
        w2 = len(class2) / image.size
        
        var1 = np.var(class1) if len(class1) > 0 else 0
        var2 = np.var(class2) if len(class2) > 0 else 0
        
        return w1 * var1 + w2 * var2

    def apply_otsu(self):
        image_uint8 = self.original_image.astype(np.uint8)
        min_val = np.min(image_uint8) + 1
        max_val = np.max(image_uint8)
        
        best_threshold = min_val
        min_variance = float('inf')
        
        for threshold in range(min_val, max_val):
            variance = self.compute_intra_class_variance(image_uint8, threshold)
            if variance < min_variance:
                min_variance = variance
                best_threshold = threshold
                
        binary_image = (image_uint8 >= best_threshold).astype(np.uint8)
        self.image = binary_image
        
        self.ax.clear()
        self.ax.imshow(binary_image, cmap='gray')
        self.ax.set_title(f"Otsu Threshold: {best_threshold}")
        self.canvas.draw()

    def avg_filter(self):
        mask = np.ones((3, 3), dtype=np.float32) / 9
        img_new = convolve(self.original_image, mask)
        self.image = img_new
        
        self.ax.clear()
        self.ax.imshow(img_new, cmap='gray')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DICOMViewer(root)
    root.mainloop()