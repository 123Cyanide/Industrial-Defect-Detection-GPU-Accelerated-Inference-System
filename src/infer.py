import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    model = Patchcore()

    ckpt = (
        "results/Patchcore/MVTecAD/"
        "bottle/v0/weights/lightning/model.ckpt"
    )

    print("Loading checkpoint:")
    print(ckpt)

    engine = Engine()

    datamodule = MVTecAD(
        root="../data",
        category="bottle",
    )


    engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt
    )


if __name__ == "__main__":
    main()
