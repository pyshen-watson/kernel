from neural_tangents import stax


class Base:

    @staticmethod
    def conv(
        out_ch,
        ksize=3,
        strides=1,
        W_std=2.0**0.5,
        b_std=0.1,
        padding="SAME",
        parameterization="ntk",
    ):
        return stax.Conv(
            out_ch,
            (ksize, ksize),
            (strides, strides),
            W_std=W_std,
            b_std=b_std,
            padding=padding,
            parameterization=parameterization,
        )

    @staticmethod
    def dense(out_dim, W_std=1.0, b_std=0.0, parameterization="ntk"):
        return stax.Dense(
            out_dim, W_std=W_std, b_std=b_std, parameterization=parameterization
        )

    @staticmethod
    def relu():
        return stax.Relu()

    @staticmethod
    def avgPool(ksize=2, strides=2):
        return stax.AvgPool((ksize, ksize), (strides, strides), "SAME")

    @staticmethod
    def globalAvgPool():
        return stax.GlobalAvgPool()

    @staticmethod
    def flatten():
        return stax.Flatten()

    @staticmethod
    def serial(layers):
        return stax.serial(*layers)
    
    @staticmethod
    def identity():
        return stax.Identity()
    
    @staticmethod
    def twoBranch(br1, br2):
        return stax.serial(
            stax.FanOut(2), 
            stax.parallel(br1, br2), 
            stax.FanInSum()
        )
