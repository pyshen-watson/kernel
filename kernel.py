import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from model import get_kernel_ntk, VGG, Resnet

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--id", type=int, default=0, help="The id of the data")
    parser.add_argument("-w", "--width", type=int, default=32, help="The width of the model")
    parser.add_argument("-l", "--level", type=int, default=1, help="The level of the model")
    parser.add_argument("-p", "--group", type=int, default=1, help="The group of the model")
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    x = np.load(f"data/IN10_image-{args.id}.npy")
    y = np.load(f"data/IN10_label-{args.id}.npy")
    size = x.shape[0]

    init_fn, apply_fn, kernel = VGG(args.width, args.level, args.group, 10)
    kernel_ntk = get_kernel_ntk(kernel)
    k_ss = np.zeros((size,size))

    with tqdm(total=size*size) as pbar:
        for i in range(size):
            for j in range(size):
                data1 = np.expand_dims(x[i], axis=0)
                data2 = np.expand_dims(x[j], axis=0)
                k_ss[i,j] = kernel_ntk(data1, data2)[0][0]
                pbar.update(1)

    lambda_K, _= np.linalg.eig(k_ss)
    print(lambda_K.min())

    with open(f"{args.level}x{args.group}_{args.id}.txt", "w") as f:
        f.write(str(lambda_K.min().item()))


