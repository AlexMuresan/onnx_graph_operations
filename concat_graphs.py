import os
import collections

import onnx
import onnx.helper as helper
import onnx.checker as checker
import onnx.optimizer as optimizer

import netron

from pprint import pprint
from collections import OrderedDict


def get_model_dict(model_path):
    '''
    Description:
        Loads a model from the given path and returns it as a dictionary
    Params:
        model_path (string): the path at which the model resides
    Returns:
        dictionary
    '''

    graph_dict = {}
    node_types = {}
    node_types = collections.OrderedDict(node_types)
    model = onnx.load(os.path.abspath(model_path))
    checker.check_model(model)

    graph = model.graph
    checker.check_graph(graph)

    nodes = graph.node

    for attr in nodes:
        node_attr = {}
        if attr.op_type in node_types:
            node_types[attr.op_type] += 1
        else:
            node_types[attr.op_type] = 0

        node_attr['op_type'] = attr.op_type
        node_attr['inputs'] = attr.input
        node_attr['outputs'] = attr.output

        for a in attr.attribute:
            node_attr[a.name] = a.i or a.ints or a.f or a.floats
        node_name = attr.op_type + '_' + str(node_types[attr.op_type])
        graph_dict[node_name] = node_attr

    return graph_dict


def get_model_params(model_path, as_dict=False, get_inputs=True,
                     get_outputs=True):
    '''
    Description:
        Loads a model from the given path and returns it's parameters
    Params:
        model_path (string): the path at which the model resides
        get_inputs (bool): get a dictionary with the inputs of the model
        get_outputs (bool): get a dictionary with the outputs of the model
    Returns:
        dictionary or dictionary pair
    '''

    model = onnx.load(os.path.abspath(model_path))
    checker.check_model(model)
    print("Model checks out!")

    if get_inputs:
        input_dims = {}
        for input in model.graph.input:
            input_name = input.name
            input_dim = input.type.tensor_type.shape.dim

            in_dims = []
            for dim in input_dim:
                in_dims.append(dim.dim_value)
            input_dims[input_name] = in_dims

    if get_outputs:
        output_dims = {}
        for output in model.graph.output:
            output_name = output.name
            output_dim = output.type.tensor_type.shape.dim

            out_dims = []
            for dim in output_dim:
                out_dims.append(dim.dim_value)
            output_dims[output_name] = out_dims

    if as_dict and get_inputs:
        input_dict = {}
        for i, key in zip(range(len(input_dims)), input_dims):
            name = 'tensor_value_info_' + str(i)
            input_dict[name] = {'name': key,
                                'shape': input_dims[key],
                                'elem_type': onnx.TensorProto.FLOAT}
        input_dims = input_dict

    if as_dict and get_outputs:
        output_dict = {}
        for i, key in zip(range(len(output_dims)), output_dims):
            name = 'tensor_value_info_' + str(i)
            output_dict[name] = {'name': key,
                                 'shape': output_dims[key],
                                 'elem_type': onnx.TensorProto.FLOAT}
        output_dims = output_dict

    if get_inputs and get_outputs:
        return input_dims, output_dims
    elif get_inputs:
        return input_dims
    else:
        return output_dims


# def check_for_collisions(graph_list):
#     keys_1 = set(graph_list[0].keys())
#     keys_2 = set(graph_list[1].keys())

#     keys_1.intersection(keys_2)


def from_dicts_to_graph(graph_name, graph_dicts, input_dicts, output_dicts):
    graph_dicts = ([graph_dicts]
                   if type(graph_dicts) != list else graph_dicts)
    input_dicts = ([input_dicts]
                   if type(input_dicts) != list else input_dicts)
    output_dicts = ([output_dicts]
                    if type(output_dicts) != list else output_dicts)

    # check_for_collisions(graph_dicts)

    graph_def = helper.make_graph(
        nodes=[
            helper.make_node(**{key: graph_dict[node][key]
                             for key in graph_dict[node]})
            for graph_dict in graph_dicts for node in graph_dict
        ],
        inputs=[
            helper.make_tensor_value_info(**{key: input_dict[tavi][key]
                                          for key in input_dict[tavi]})
            for input_dict in input_dicts for tavi in input_dict
        ],
        outputs=[
            helper.make_tensor_value_info(**{key: output_dict[tavi][key]
                                          for key in output_dict[tavi]})
            for output_dict in output_dicts for tavi in output_dict
        ],
        name=graph_name)

    onnx.checker.check_graph(graph_def)
    print("Graph checks out!")

    return graph_def


graph_1 = get_model_dict('./model_1.onnx')
graph_2 = get_model_dict('./model_2.onnx')


model_inputs_1, model_outputs_1 = get_model_params(
    './model_1.onnx',
    as_dict=True)


model_inputs_2, model_outputs_2 = get_model_params(
    './model_2.onnx',
    as_dict=True)


# graph_2['Conv_0']['inputs'] = graph_1['Conv_0']['inputs']
# del model_inputs_2['tensor_value_info_0']
# del model_inputs_2['tensor_value_info_1']

graph_2['BatchNormalization_0']['inputs'][0] = graph_1['BatchNormalization_0']['inputs'][0]
print(graph_2['BatchNormalization_0']['inputs'])

del graph_2['Conv_0']
del model_inputs_2['tensor_value_info_0']
del model_inputs_2['tensor_value_info_1']

graph_def = from_dicts_to_graph("Reconstructed", [graph_1, graph_2],
                                [model_inputs_1, model_inputs_2],
                                [model_outputs_1, model_outputs_2])

model_def = helper.make_model(graph_def)
print("Model 2 checks out!")

onnx.save_model(model_def, 'combined_model_2.onnx')
# netron.start('combined_model_2.onnx')
