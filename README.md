# onnx_graph_operations
Convert onnx models and graphs into dictionaries and do various operations on them

## Capabilities
The concat_graphs.py script can do the following:
* convert a .onnx file into a dictionary
* grab the inputs and outputs of a .onnx file as a dictionary or as a list
* convert a dictionary or a list of dictionaries containing model specifications, inputs and outputs into a onnx graph

The create_onnx_test_graphs.py generates two onnx models that can be used with concat_graphs.py in order to test functionality.
This can be edited in order to create different architectures.

## TO-DO
* find a way to programatically combine and modify onnx models and not manually edit the dictionaries:
	* check for collisions in the dictionaries that will be combined (onnx will throw an error if something is used as an output for two different things)
	* find a way to specify at which node should the graphs be merged and automatically identify what should be added and what should be removed so the thing will work properly
* any other things that show up and will prevent this for working properly
