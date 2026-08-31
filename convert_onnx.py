import tensorflow as tf
import tf2onnx
# আপনার বর্তমান main.py থেকে মডেল ক্লাসগুলো ইম্পোর্ট করা হচ্ছে
from main import JointConvNeXtModel 

print("Loading model...")
model = JointConvNeXtModel(num_classes=4)
model(tf.zeros((1, 224, 224, 3)), training=False)
model.load_weights("best_joint_convnext_tiny.weights.h5")

print("Converting to ONNX...")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open("skin_lesion_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ Conversion complete! File saved as 'skin_lesion_model.onnx'")