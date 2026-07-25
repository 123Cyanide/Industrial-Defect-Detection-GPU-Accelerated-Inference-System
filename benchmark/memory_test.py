import torch
from anomalib.models import Patchcore


device="cuda"


ckpt = (
"/home/caojf/projects/industrial-ai/"
"results/Patchcore/MVTecAD/bottle/v0/"
"weights/lightning/model.ckpt"
)


print("Loading checkpoint...")


model = Patchcore.load_from_checkpoint(
    ckpt
)

model.to(device)
model.half()
model.eval()


x=torch.randn(
    1,3,256,256,
    device=device
).half()


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


print("Inference...")


with torch.no_grad():

    for i in range(10):
        output=model(x)


torch.cuda.synchronize()


print("===================")

print(
"Allocated:",
torch.cuda.memory_allocated()/1024**2,
"MB"
)

print(
"Reserved:",
torch.cuda.memory_reserved()/1024**2,
"MB"
)


print(
"Peak:",
torch.cuda.max_memory_allocated()/1024**2,
"MB"
)

print("===================")
