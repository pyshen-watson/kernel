from .base import Base


def ResnetBlock(out_ch, io_mismatch=False):

    mainPath = Base.serial(
        [
            Base.relu(),
            Base.conv(out_ch),
            Base.relu(),
            Base.conv(out_ch),
        ]
    )

    shortcut = Base.conv(out_ch, ksize=1) if io_mismatch else Base.identity()
    return Base.twoBranch(mainPath, shortcut)


def ResnetGroup(n, out_ch):
    layers = [ResnetBlock(out_ch, io_mismatch=True)]
    layers += [ResnetBlock(out_ch) for _ in range(n - 1)]
    layers += [Base.avgPool()]
    return Base.serial(layers)


def Resnet(width_base=32, level=1, pooling=1, n_class=10):

    layers = [Base.conv(width_base, 7, 2), Base.avgPool(ksize=3)]

    for _ in range(pooling - 1):
        layers += [ResnetGroup(level, width_base)]
        width_base *= 2

    layers += [Base.globalAvgPool(), Base.flatten(), Base.dense(n_class)]
    return Base.serial(layers)
