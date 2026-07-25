import torch
import numpy as np
import time

from PIL import Image
from torchvision import transforms

from anomalib.models import Patchcore


DEVICE = "cuda"


IMAGE_PATH = (
    "/home/caojf/projects/data/"
    "bottle/test/contamination/015.png"
)


CHECKPOINT = (
    "results/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)


WARMUP = 50
ITERATIONS = 200



def load_model():

    model = Patchcore()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cuda"
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    return model



def load_image():

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    x = transform(image)

    x = x.unsqueeze(0)

    return x.to(DEVICE)



def benchmark(model, x):

    print("Warmup...")

    with torch.no_grad():

        for _ in range(WARMUP):
            model(x)


    torch.cuda.synchronize()


    print("Benchmark...")


    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )


    times=[]


    with torch.no_grad():

        for _ in range(ITERATIONS):

            start.record()

            model(x)

            end.record()


            torch.cuda.synchronize()


            times.append(
                start.elapsed_time(end)
            )


    times=np.array(times)


    print("============================")
    print("PatchCore FP32 Benchmark")
    print("============================")

    print(
        f"mean: {times.mean():.3f} ms"
    )

    print(
        f"p50: {np.percentile(times,50):.3f} ms"
    )

    print(
        f"p95: {np.percentile(times,95):.3f} ms"
    )

    print(
        f"FPS: {1000/times.mean():.2f}"
    )

    print("============================")



if __name__=="__main__":

    model=load_model()

    image=load_image()

    benchmark(
        model,
        image
    )
