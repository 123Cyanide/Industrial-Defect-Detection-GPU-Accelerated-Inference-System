import torch
from anomalib.models.components.feature_extractors import TimmFeatureExtractor


device="cuda"


# 和 PatchCore 一致
extractor = TimmFeatureExtractor(
    backbone="wide_resnet50_2",
    pre_trained=True,
    layers=[
        "layer2",
        "layer3"
    ]
)

extractor.eval()
extractor.to(device)


dummy = torch.randn(
    1,
    3,
    256,
    256,
    device=device
)


import os; os.makedirs("models", exist_ok=True)

torch.onnx.export(
    extractor,
    dummy,
    "models/patchcore_feature_extractor.onnx",
    opset_version=17,
    input_names=[
        "image"
    ],
    output_names=[
        "features"
    ],
    dynamic_axes={
        "image":{
            0:"batch"
        },
       "features":{
           0:"batch"        
        }
    }
)


print("ONNX export finished")
