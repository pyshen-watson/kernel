import random
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument( "-r", "--root_dir", type=str, default="../data/ImageNet10/train", help="The path to the root directory of the dataset", )
    parser.add_argument( "-s", "--size", type=int, default=112, help="The size of the image" ) 
    parser.add_argument( "-n", "--num_sample", type=int, default=10, help="The number of samples per class", )
    return parser.parse_args()


def get_transform(size: int):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def transform(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - MEAN) / STD
        return img

    return transform


def sampling(root: Path, n_sample: int, transform):

    image_list = []
    label_list = []
    n_class = len(list(root.iterdir()))

    for class_dir in root.iterdir():

        # Sample images from each class
        image_paths = list(class_dir.glob("*.jpg"))
        sample_paths = random.sample(image_paths, n_sample)

        # Open in PIL.Image and do the preprocessing
        images = [Image.open(img) for img in sample_paths]
        images = [transform(img) for img in images]

        # One-hot encoding for the label
        label = np.zeros(n_class, dtype=np.float32)
        label[int(class_dir.name)] = 1.0
        labels = [label] * n_sample

        # Append to the list
        image_list.extend(images)
        label_list.extend(labels)

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
    root = Path(args.root_dir)
    transform = get_transform(args.size)  # In our case, resize to 112 x 112
    result = sampling(root, args.num_sample, transform)
    save_result(result)