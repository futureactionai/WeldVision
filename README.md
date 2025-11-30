# WeldVision — Weld Seam Line Detection System

WeldVision is an intelligent computer-vision system developed within **Future Action AI** to automatically detect welding seam lines from images and provide precise line coordinates for robotic or inspection workflows.

The system uses:
- **YOLO-based segmentation**
- **Custom Python processing blocks in AugeLab Studio**
- **Contour, skeleton, and line-fitting algorithms**
- **Automated coordinate extraction**

WeldVision was designed to support manufacturing, automation, and industrial inspection processes by providing accurate weld seam detection that can be integrated into welding robots, quality control systems, and smart factories.

---

## 🚀 Features

### **1. Weld Seam Segmentation (YOLO Model)**
- A custom-trained YOLO segmentation model detects the weld seam region.
- Outputs a segmentation mask for further geometric processing.
- Model files:
  - `best.pt` → Best-performing model during training
  - `last.pt` → Model from final training epoch

### **2. Custom “Weld Seam Line Detector” Block**
A custom AugeLab block performs:
- Mask extraction
- Largest-instance selection
- Skeletonization
- FitLine-based axis estimation
- Coordinate extraction for seam endpoints
- Annotated output image generation

### **3. End-to-End Image Processing Pipeline**
- Input: Raw metal/welding area image
- Output:  
  - Annotated image (green seam line)
  - JSON coordinate data:  
    ```json
    { "x1": ..., "y1": ..., "x2": ..., "y2": ... }
    ```

### **4. Flexible Integration**
- Works inside AugeLab Studio
- Exportable to Python scripts
- Useful for robotics, inspection, smart factories

---

## 📂 Project Structure

```
WeldVision/
│
├── models/
│   ├── best.pt        # Best YOLO model
│   └── last.pt        # Final YOLO model
│
├── custom_blocks/
│   └── WeldSeamLineFromMask.py   # Custom block source code
│
├── example_images/
│   └── sample1.jpg
│
└── README.md
```

---

## 🧩 Installation & Setup (AugeLab Studio)

### **Custom Block Installation**
Place this file into:
```
Windows:
C:\Users\<USERNAME>\AppData\\Roaming\AugeLab Studio\marketplace\custom_blocks\

```

AugeLab will auto-detect it after restarting.

### **Model Installation**
Place `best.pt` and `last.pt` into:
```
C:\Users\<USERNAME>\AugeLab Studio\models\
```

Update the model path inside the block:
```python
MODEL_PATH = r"C:\\Users\\<USERNAME>\\AugeLab Studio\\models\\best.pt"
```

---

## 🧠 How It Works

### **1. YOLO Segmentation**
The model predicts a segmentation mask representing the weld seam region.

### **2. Mask Selection**
If multiple mask instances exist, the system selects the one with the largest pixel area.

### **3. Skeletonization**
The mask is thinned to a 1-pixel width centerline to reduce noise from wide segmentation regions.

### **4. Line Fitting**
OpenCV's `fitLine` computes a best-fit line along the weld seam.

### **5. Endpoint Extraction**
All points on the skeleton are projected onto this line to find the true endpoints.

### **6. Output Generation**
- A green line representing the seam is drawn.
- Coordinates are exported in JSON.

---

## 🛠️ Example Output

A few real examples from the WeldVision system detecting weld seam lines:

<p align="center">
  <img src="images/output1.jpg" width="600">
</p>

<p align="center">
  <img src="images/output2.jpg" width="600">
</p>

*Green line represents the automatically detected weld seam axis.*

---

## 📌 Known Limitations

- Diagonally oriented weld seams can produce wider YOLO segmentation, requiring centerline extraction.
- Model accuracy depends heavily on dataset labeling quality.
- Reflective or low-contrast surfaces may require additional dataset augmentation.

---

## 🧮 Future Improvements

- Line-stability enhancement using median centerline extraction  
- Welding anomaly detection (porosity, gaps, defects)  
- Fully automated robotic weld-path generation  
- Multi-angle 3D seam reconstruction  
- Integrated YOLO + classical line detection hybrid pipeline  

---

## 🤝 Contributions

This project is developed under **Future Action AI** as part of the community’s industrial computer-vision initiatives.

Contributions, pull requests, and feature additions are welcome!

---

## 📝 License

MIT License — free for personal and commercial use.

---

## 📦 Dataset

The training dataset for WeldVision was obtained from [Roboflow Universe — Weld Seam Segmentation2](https://universe.roboflow.com/weld-q7cmf/weld-seam-segmentation2).  
The dataset is used under the license specified on the Roboflow page (see license info on dataset page).  
Please refer to the original dataset for full license and attribution details.

---

## 👩‍💻 Authors

- **Future Action AI Community** — Research & Development Support
