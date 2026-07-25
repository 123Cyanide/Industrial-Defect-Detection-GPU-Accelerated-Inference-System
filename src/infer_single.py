import torch
from PIL import Image
from torchvision import transforms

from anomalib.models import Patchcore
from anomalib.data.utils import read_image


DEVICE = "cuda"


IMAGE_PATH = (
    "/home/caojf/projects/data/"
    "bottle/test/contamination/015.png"
)


CHECKPOINT = (
    "results/Patchcore/MVTecAD/"
    "bottle/v0/weights/lightning/model.ckpt"
)


def main():

    print("Loading model...")

    model = Patchcore()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cuda"
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False
    )

    print("======================")
    print("Missing keys:")
    print(missing)

    print("Unexpected keys:")
    print(unexpected)
    print("======================")    


    model.to(DEVICE)
    model.eval()


    print("Loading image...")

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")


    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])


    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)


    print(input_tensor.shape)


    with torch.no_grad():

        output = model(
            input_tensor
        )


    print(output)



if __name__ == "__main__":
    main()
