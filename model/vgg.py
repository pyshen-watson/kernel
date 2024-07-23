from .base import Base

def VGGBlock(out_ch, num_conv):
    layers = []
    for _ in range(num_conv):
        layers += [Base.conv(out_ch), Base.relu()]
    layers += [Base.avgPool()]
    return Base.serial(layers)

def VGG(width_Base=32, level=2, pooling=3, n_class=10):
    
    layers = []
    out_ch = width_Base

    # Convolutional backbone
    for _ in range(pooling):
        layers += [VGGBlock(out_ch, level)]
        out_ch *= 2

    # Fully connected layers
    layers += [
        Base.globalAvgPool(),
        Base.flatten(),
        Base.dense(n_class),
    ]

    return Base.serial(layers)
