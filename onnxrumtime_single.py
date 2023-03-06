from onnxruntime import InferenceSession
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F

"""
This code for single image inference from onnx model
"""


def preprocessing(img, size):
    mean = [0.486, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = img.resize(size, Image.LANCZOS)
    img = F.to_tensor(img)
    img = F.normalize(img, mean, std)
    img = img.numpy()
    img = np.expand_dims(img, 0)
    return img


def load_img(image_path):
    return Image.open(image_path).convert("RGB")


if __name__ == "__main__":
    sess = InferenceSession("efficientnet-b3.onnx")
    input_name = sess.get_inputs()[0].name
    image_path = "sample2.jpg"
    PIL_img = load_img(image_path)
    input_img = preprocessing(PIL_img, (320,320))
    result = sess.run(None, {input_name: input_img})
    print(result)
