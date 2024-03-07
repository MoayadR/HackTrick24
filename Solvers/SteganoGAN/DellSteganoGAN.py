from utils import decode
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def stegano_solver(cover_im: np.ndarray, message: str) -> str:
    ## add your code here
    pass

def decode_image(encoded_im: np.ndarray) -> str:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    tensor_image = transform(encoded_im)

    tensor_image = tensor_image.unsqueeze(0)

    decoded_text = decode(tensor_image)
    return decoded_text

img = Image.open('sample_example/encoded.png')
decoded_image = decode_image(np.array(img))
print(decoded_image)