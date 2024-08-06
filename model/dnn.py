from .base import Base

def DNN(width_Base=2048, level=1, n_class=10):
    
    layers = [Base.flatten()]

    for i in range(level):
        if i == level - 1:
            layers += [Base.dense(n_class)]
        else:
            layers += [Base.dense(width_Base), Base.relu()]

    return Base.serial(layers)
