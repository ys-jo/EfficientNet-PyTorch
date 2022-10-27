from onnxruntime import InferenceSession
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import torchvision
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


"""
This code for ImageNet inference from onnx model
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
    sess = InferenceSession("test-b3.onnx")
    input_name = sess.get_inputs()[0].name

    t = []
    mean = [0.486, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t.extend([transforms.Resize((320,320)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    t = transforms.Compose(t)
    dataset = torchvision.datasets.ImageNet(root="/home/ysjo/dataset/dataset/ImageNet/",split='val', transform=t)
    class_names = os.walk("/home/ysjo/dataset/dataset/ImageNet/val").__next__()[1]
    class_names.sort()
    dataloader = DataLoader(dataset,
                        pin_memory=True,
                        batch_size=1,
                        num_workers=12)
    cnt = 0
    correct = 0

    for i, batch in enumerate(tqdm(dataloader)):
        x, y = batch
        x = x.numpy()
        y = y.numpy()
        #x = np.expand_dims(x, 0)
        result = sess.run(None, {input_name: x})
        cnt +=1
        if result == y:
            correct +=1
    print(correct/cnt)
