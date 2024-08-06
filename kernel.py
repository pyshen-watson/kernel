import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from model import get_kernel_ntk, DNN, VGG, Resnet

import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="data/IN10x10", help="The dataset")
    parser.add_argument("-i", "--id", type=int, default=0, help="The id of the data")

    parser.add_argument("-m", "--model", type=str, choices=["DNN", "CNN", "VGG", "ResNet"], help="The width of the model")
    parser.add_argument("-l", "--level", type=int, default=1, help="The level of the model")
    parser.add_argument("-g", "--group", type=int, default=1, help="The group of the model")
    return parser.parse_args()

def get_ntk(args):

    if args.model == "DNN":
        init_fn, apply_fn, kernel = DNN(2048, args.level)

    elif args.model == "CNN":
        init_fn, apply_fn, kernel = VGG(512, args.level, 1)
        
    elif args.model == "VGG":
        init_fn, apply_fn, kernel = VGG(32, args.level, args.group)
    
    elif args.model == "ResNet":
        init_fn, apply_fn, kernel = Resnet(32, args.level, args.group)
        
    else:
        raise ValueError("Model not found")
    
    return get_kernel_ntk(kernel)


if __name__ == '__main__':

    args = get_args()
    data_path = Path(args.data_dir) / f"image-{args.id}.npy"
    x = np.load(data_path)
    size = x.shape[0] # Number of samples
    
    kernel_ntk = get_ntk(args)
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

    with open(f"output/{args.level}x{args.group}_{args.id}.txt", "w") as f:
        f.write(str(lambda_K.min().item()))


