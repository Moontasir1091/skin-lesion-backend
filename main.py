from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ConvNeXtTiny
import numpy as np
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

def double_conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_real_unet(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)
    c1 = double_conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = double_conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = double_conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = double_conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = double_conv_block(p4, 512)
    
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = double_conv_block(u6, 256)
    
    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = double_conv_block(u7, 128)
    
    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = double_conv_block(u8, 64)
    
    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = double_conv_block(u9, 32)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid', name="segmentation_output")(c9)
    return models.Model(inputs=inputs, outputs=outputs, name="RealUNet")

class JointConvNeXtModel(tf.keras.Model):
    def __init__(self, num_classes=4):
        super(JointConvNeXtModel, self).__init__()
        self.unet = build_real_unet()
        self.convnext = ConvNeXtTiny(include_top=False, weights=None, input_shape=(224, 224, 3))
        self.head = models.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, dtype='float32', name="classification_output")
        ])

    def call(self, inputs, training=False):
        pred_mask = self.unet(inputs, training=training)
        factor = 0.3 + 0.7 * tf.cast(pred_mask, inputs.dtype)
        masked_image_raw = inputs * factor
        
        mean = tf.constant([0.485, 0.456, 0.406], dtype=inputs.dtype)
        std = tf.constant([0.229, 0.224, 0.225], dtype=inputs.dtype)
        normalized_input = (masked_image_raw - mean) / std
        
        feats = self.convnext(normalized_input, training=training)
        class_output = self.head(feats, training=training)
        return pred_mask, masked_image_raw, class_output

print("Loading model weights...")
model = JointConvNeXtModel(num_classes=4)
model(tf.zeros((1, 224, 224, 3)), training=False)
model.load_weights("best_joint_convnext_tiny.weights.h5")
print("Model loaded successfully!")

CLASS_MAP = {0: "Chickenpox", 1: "Measles", 2: "Monkeypox", 3: "Healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize((224, 224), Image.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = tf.expand_dims(img_arr, 0)

        pred_mask, _, class_outputs = model(img_tensor, training=False)
        probabilities = tf.nn.softmax(class_outputs, axis=1).numpy()[0]
        pred_idx = np.argmax(probabilities)
        
        mask_np = pred_mask.numpy()[0, :, :, 0]
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