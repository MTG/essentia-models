import onnxruntime as ort
import numpy as np

# Load ONNX model (replace with your model file)
model_path = "identity2x2.onnx"
session = ort.InferenceSession(model_path)

# Show input/output names
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]
print("Inputs:", input_names)
print("Outputs:", output_names)

# Create two 3x3 input arrays
input1 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32)

input2 = np.array([
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
], dtype=np.float32)

# prepare model inputs and batches
input1, input2 = (np.float32(np.random.random((3, 3))) for _ in range(2))

print("input1: ")
print(input1)
print("input2: ")
print(input2)

# Run inference
print("Running inference...")
outputs = session.run(
    output_names,                 # which outputs to fetch
    {
        input_names[0]: input1,   # first input
        input_names[1]: input2    # second input
    }
)

# Print results
for i, out in enumerate(outputs):
    print(f"Output {i}:")
    print(out)
