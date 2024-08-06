import random
import numpy as np
from PIL import Image
from typing import List
from argparse import ArgumentParser
from torchvision.datasets import CIFAR10


def get_args():
    parser = ArgumentParser()
    parser.add_argument( "-r", "--root_dir", type=str, default="../data/", help="The path to the root directory of the dataset", )
    parser.add_argument( "-n", "--num_sample", type=int, default=10, help="The number of samples per class", )
    return parser.parse_args()


def get_transform():

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def transform(img: np.ndarray) -> np.ndarray:
        img = (img / 255.0 - MEAN) / STD
        return img

    return transform


def sampling(images: np.ndarray, labels: List[int], n_sample: int, transform):

    image_list = []
    label_list = []
    n_class = max(labels) + 1
    labels = np.array(labels)

    for class_id in range(n_class):

        class_index = np.where(labels == class_id)[0]
        sample_index = np.random.choice(class_index, n_sample, replace=False)

        # Sample images from each class
        sample_images = images[sample_index]
        image_list += [transform(img) for img in sample_images]

        # One-hot encoding for the label
        sample_label = np.zeros(n_class, dtype=np.float32)
        sample_label[class_id] = 1.0
        label_list += [sample_label] * n_sample


    return {
        "image": np.stack(image_list, axis=0),
        "label": np.stack(label_list, axis=0),
    }


def save_result(result: dict):

    for key, value in result.items():
        np.save(f"{key}.npy", value)
        print(f"Save: {key}.npy\tShape: {value.shape}")

if __name__ == "__main__":

    args = get_args()
    ds = CIFAR10(args.root_dir, train=True, download=True)
    images, labels = ds.data, ds.targets
    transform = get_transform()
    result = sampling(images, labels, args.num_sample, transform)
    save_result(result)