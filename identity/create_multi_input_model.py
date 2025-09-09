import onnx
from onnx import helper, TensorProto

# Define inputs
input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [None, 3])
input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [None, 3])

# Define outputs (same shape as inputs)
output1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [None, 3])
output2 = helper.make_tensor_value_info("output2", TensorProto.FLOAT, [None, 3])

# Create identity nodes (one for each input→output)
node1 = helper.make_node(
    "Identity",  # Operator
    inputs=["input1"],
    outputs=["output1"],
)
node2 = helper.make_node("Identity", inputs=["input2"], outputs=["output2"])

# Build the graph
graph_def = helper.make_graph(
    [node1, node2], "identity_graph", [input1, input2], [output1, output2]
)

# Create the model
model_def = helper.make_model(graph_def, producer_name="onnx-example")

# Save the model
onnx.save(model_def, "identity2x2.onnx")

print("Model saved as identity2x2.onnx")

