import onnxruntime as ort
import numpy as np
import cv2
import time

# Path to your ONNX model
onnx_model_path = ".src/yolov8.onnx"

# Create session with QNN EP
so = ort.SessionOptions()
so.enable_profiling = True   # To enable ONNX Runtime internal profiling
# Optionally disable fallback so you ensure model runs on NPU as much as possible
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

# Provider options (you may need to specify backend path or settings depending on device)
provider_options = [
    {
        "backend_type": "htp",
    }
]

session = ort.InferenceSession(onnx_model_path, sess_options=so,
                                 providers=["QNNExecutionProvider"],
                                 provider_options=provider_options)

print("Available providers:", ort.get_available_providers())
print("Session providers being used:", session.get_providers())
