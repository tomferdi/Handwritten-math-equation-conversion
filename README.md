# Handwritten Math Equation Conversion

This project converts images of handwritten math equations into LaTeX format using a pre-trained ONNX model. It uses a Streamlit interface to load and display the input image and show the predicted LaTeX output.

---

## 📦 Requirements

- **Python version**: 3.13  
- Python dependencies:

You can install all required packages with:

```bash
pip install -r requirements.txt
If requirements.txt is missing, install manually:

pip install streamlit onnxruntime pillow matplotlib numpy
🚀 How to Run the App
Clone the Repository

git clone https://github.com/tomferdi/Handwritten-math-equation-conversion
cd Handwritten-math-equation-conversion
(Optional) Download the ONNX Model
To enable prediction, you need a file named model.onnx.

This file is not included in the repository.

If available, place it in the root directory of the project (same folder as app.py).

If you don't include model.onnx, the app will still open but cannot perform prediction.

Add Input Images
Place handwritten math equation images inside the test_images/ folder.

You can edit the file path inside app.py if you want to change which image is loaded:


img_path = "test_images/your_image.png"
Note: The model expects grayscale images.
The exact image dimensions required by the model are not documented, but using a square image (like 256x256 pixels) is a safe choice.

Run the Streamlit App

streamlit run app.py


