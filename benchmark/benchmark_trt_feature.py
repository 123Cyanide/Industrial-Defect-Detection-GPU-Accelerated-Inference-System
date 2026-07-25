import onnxruntime as ort
import numpy as np
import time


model = "/home/caojf/projects/industrial-ai/patchcore_feature_extractor.onnx"


session = ort.InferenceSession(
    model,
    providers=[
        "TensorrtExecutionProvider"
    ]
)

print("Providers:")
print(session.get_providers())


input_name = session.get_inputs()[0].name


x = np.random.randn(
    1,3,256,256
).astype(np.float32)


# warmup
print("Warmup...")
for _ in range(50):
    session.run(
        None,
        {input_name:x}
    )


print("Benchmark...")

times=[]

for _ in range(200):

    start=time.perf_counter()

    session.run(
        None,
        {input_name:x}
    )

    end=time.perf_counter()

    times.append(
        (end-start)*1000
    )


print("==========================")
print("TensorRT Feature Extractor")
print("==========================")

print(
    "mean:",
    np.mean(times),
    "ms"
)

print(
    "p50:",
    np.percentile(times,50),
    "ms"
)

print(
    "p95:",
    np.percentile(times,95),
    "ms"
)

print(
    "FPS:",
    1000/np.mean(times)
)
