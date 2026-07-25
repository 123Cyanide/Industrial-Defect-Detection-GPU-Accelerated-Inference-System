from anomalib.data import MVTecAD
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


def main():

    datamodule = MVTecAD(
        root="../data",
        category="bottle",
        train_batch_size=4,
        eval_batch_size=1,
    )

    model = Patchcore()

    engine = Engine()

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="results/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt",
    )


if __name__ == "__main__":
    main()
