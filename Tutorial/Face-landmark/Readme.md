# Facial Landmark Detection

This project detects facial landmarks using dlib's 68-point face landmark predictor.

## 📜 Requirements

Make sure you have the following installed:

- **Python 3.8+**
- **dlib**
- **OpenCV (`cv2`)**
- **imutils**
- **NumPy**

You can install the required dependencies using:

```sh
pip install -r requirements.txt
```

### **Alternatively, install manually:**
```sh
pip install dlib opencv-python imutils numpy
```

## 🚀 How to Run

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/invi-bhagyesh/facial-landmark-detection.git
   cd facial-landmark-detection
   ```

2. **Download the Shape Predictor Model:**
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib’s model download page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) or by running
   ```sh
   python etch_dlib_model.py
   ```
   - Extract and place it in the project folder.

3. **Run the Facial Landmark Detection Script:**
   ```sh
   python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/karina.jpeg
   ```

## 📜 License

This project is open-source and free to use.

---

