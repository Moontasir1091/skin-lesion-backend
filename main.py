from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import io
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading ONNX model...")
# ONNX Runtime সেশন চালু করা
ort_session = ort.InferenceSession("skin_lesion_model.onnx")
input_name = ort_session.get_inputs()[0].name
print("Model loaded successfully!")

CLASS_MAP = {0: "Chickenpox", 1: "Measles", 2: "Monkeypox", 3: "Healthy"}

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize((224, 224), Image.BILINEAR)
        
        # NumPy ব্যবহার করে প্রি-প্রসেসিং
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_arr, 0)

        # ONNX দিয়ে প্রেডিকশন
        ort_outs = ort_session.run(None, {input_name: img_tensor})
        
        pred_mask = ort_outs[0]
        class_outputs = ort_outs[-1] 
        
        probabilities = softmax(class_outputs)[0]
        pred_idx = np.argmax(probabilities)
        
        # PIL দিয়ে হিটম্যাপ তৈরি
        mask_np = pred_mask[0, :, :, 0]
        mask_uint8 = np.uint8(255 * mask_np)
        
        mask_pil = Image.fromarray(mask_uint8, mode='L')
        heatmap_pil = ImageOps.colorize(mask_pil, black="blue", mid="yellow", white="red")
        
        orig_rgba = img_resized.convert("RGBA")
        heatmap_rgba = heatmap_pil.convert("RGBA")
        blended = Image.blend(orig_rgba, heatmap_rgba, alpha=0.4)
        
        blended_rgb = blended.convert("RGB")
        buffered = io.BytesIO()
        blended_rgb.save(buffered, format="JPEG")
        heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {
            "success": True,
            "disease": CLASS_MAP[pred_idx],
            "confidence": f"{probabilities[pred_idx]*100:.2f}%",
            "gradcam_base64": heatmap_b64
        }
    except Exception as e:
        return {"success": False, "error": str(e)}