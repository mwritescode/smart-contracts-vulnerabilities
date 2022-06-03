import math
import numpy as np
from PIL import Image
from hexbytes import HexBytes

SAFE_IDX = 4

def __get_one_hot_encoded_label(label):
    one_hot = np.zeros(5)
    for elem in label:
        if elem < SAFE_IDX:
            one_hot[elem] = 1
        elif elem > SAFE_IDX:
            one_hot[elem-1] = 1
    return one_hot

def __get_RGB_image(bytecode):
    image = np.frombuffer(bytecode, dtype=np.uint8)
    sqrt_len = int(math.ceil(math.sqrt(len(image))))
    image = np.pad(image, pad_width=(0, sqrt_len**2 - len(image)))
    image = np.reshape(image, (sqrt_len, sqrt_len))
    image = Image.fromarray(image).convert('RGB')
    return image

def generate_image_and_label(example):
    code = HexBytes(example['bytecode'])
    example['image'] = __get_RGB_image(code)
    example['label'] = __get_one_hot_encoded_label(example['slither'])
    return example

def generate_image_and_binary_label(example):
    code = HexBytes(example['bytecode'])
    example['image'] = __get_RGB_image(code)
    example['label'] = 0.0 if SAFE_IDX in example['slither'] else 1.0
    return example

def generate_signal_and_label(example):
    code = HexBytes(example['bytecode'])
    image = np.frombuffer(code, dtype=np.uint8)
    example['image'] = image
    example['label'] = __get_one_hot_encoded_label(example['slither'])
    return example

def generate_signal_and_binary_label(example):
    code = HexBytes(example['bytecode'])
    image = np.frombuffer(code, dtype=np.uint8)
    example['image'] = image
    example['label'] =  0.0 if SAFE_IDX in example['slither'] else 1.0
    return example
