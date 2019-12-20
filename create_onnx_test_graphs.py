import onnx
import onnx.optimizer as optimizer
import onnx.helper as helper

from onnx.tools import update_model_dims
from onnx import AttributeProto, TensorProto, GraphProto

graph_def_1 = helper.make_graph(
    nodes=[
        helper.make_node('Conv', inputs=['X', 'W1'], outputs=['o1'],
                         strides=[2, 2], pads=[1, 1, 1, 1],
                         kernel_shape=[7, 7]),
        helper.make_node('BatchNormalization',
                         inputs=['o1', 'scale1', 'bias1', 'mean1', 'var1'],
                         outputs=['o2'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o2'], outputs=['o3']),
        helper.make_node('MaxPool', inputs=['o3'], outputs=['o4', 'o5'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[2, 2]),

        helper.make_node('Conv', inputs=['o4', 'W2'], outputs=['o6'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o6', 'scale2', 'bias2', 'mean2', 'var2'],
                         outputs=['o7'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o7'], outputs=['o8']),

        helper.make_node('Conv', inputs=['o8', 'W3'], outputs=['o9'],
                         strides=[1, 1], pads=[1, 1, 1, 1],
                         kernel_shape=[3, 3], group=32),
        helper.make_node('BatchNormalization',
                         inputs=['o9', 'scale3', 'bias3', 'mean3', 'var3'],
                         outputs=['o10'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o10'], outputs=['o11']),

        helper.make_node('Conv', inputs=['o11', 'W4'], outputs=['o12'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o12', 'scale4', 'bias4', 'mean4', 'var4'],
                         outputs=['o13'], epsilon=1e-5, momentum=0.001),

        helper.make_node('Conv', inputs=['o5', 'W5'], outputs=['o14'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o14', 'scale5', 'bias5', 'mean5', 'var5'],
                         outputs=['o15'], epsilon=1e-5, momentum=0.001),

        helper.make_node('Add', inputs=['o13', 'o15'], outputs=['o16']),
        helper.make_node('Relu', inputs=['o16'], outputs=['sum_1']),
    ],
    name='ns_gone_wild_onnx_6',
    inputs=[
        helper.make_tensor_value_info(name='X', shape=[10, 3, 224, 224],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W1', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W3', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W4', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W5', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale1', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale3', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale4', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale5', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias1', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias3', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias4', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias5', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean1', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean3', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean4', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean5', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var1', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var3', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var4', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var5', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
    ],
    outputs=[
        helper.make_tensor_value_info(name='sum_1', shape=[66, 66],
                                      elem_type=onnx.TensorProto.FLOAT),
    ]
)

graph_def_2 = helper.make_graph(
    nodes=[
        helper.make_node('Conv', inputs=['X_2', 'W1_2'], outputs=['o1_2'],
                         strides=[2, 2], pads=[1, 1, 1, 1],
                         kernel_shape=[7, 7]),
        helper.make_node('BatchNormalization',
                         inputs=['o1_2', 'scale1_2', 'bias1_2', 'mean1_2', 'var1_2'],
                         outputs=['o2_2'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o2_2'], outputs=['o3_2']),
        helper.make_node('MaxPool', inputs=['o3_2'], outputs=['o4_2', 'o5_2'],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                         strides=[2, 2]),

        helper.make_node('Conv', inputs=['o4_2', 'W2_2'], outputs=['o6_2'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o6_2', 'scale2_2', 'bias2_2', 'mean2_2', 'var2_2'],
                         outputs=['o7_2'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o7_2'], outputs=['o8_2']),

        helper.make_node('Conv', inputs=['o8_2', 'W3_2'], outputs=['o9_2'],
                         strides=[1, 1], pads=[1, 1, 1, 1],
                         kernel_shape=[3, 3], group=32),
        helper.make_node('BatchNormalization',
                         inputs=['o9_2', 'scale3_2', 'bias3_2', 'mean3_2', 'var3_2'],
                         outputs=['o10_2'], epsilon=1e-5, momentum=0.001),
        helper.make_node('Relu', inputs=['o10_2'], outputs=['o11_2']),

        helper.make_node('Conv', inputs=['o11_2', 'W4_2'], outputs=['o12_2'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o12_2', 'scale4_2', 'bias4_2', 'mean4_2', 'var4_2'],
                         outputs=['o13_2'], epsilon=1e-5, momentum=0.001),

        helper.make_node('Conv', inputs=['o5_2', 'W5_2'], outputs=['o14_2'],
                         strides=[1, 1], pads=[0, 0, 0, 0],
                         kernel_shape=[1, 1]),
        helper.make_node('BatchNormalization',
                         inputs=['o14_2', 'scale5_2', 'bias5_2', 'mean5_2', 'var5_2'],
                         outputs=['o15_2'], epsilon=1e-5, momentum=0.001),

        helper.make_node('Add', inputs=['o13_2', 'o15_2'], outputs=['o16_2']),
        helper.make_node('Relu', inputs=['o16_2'], outputs=['sum_2']),
    ],
    name='ns_gone_wild_onnx_7',
    inputs=[
        helper.make_tensor_value_info(name='X_2', shape=[10, 3, 224, 224],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W1_2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W2_2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W3_2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W4_2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='W5_2', shape=[7, 7],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale1_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale2_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale3_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale4_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='scale5_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias1_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias2_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias3_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias4_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='bias5_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean1_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean2_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean3_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean4_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='mean5_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var1_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var2_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var3_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var4_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
        helper.make_tensor_value_info(name='var5_2', shape=[1],
                                      elem_type=onnx.TensorProto.FLOAT),
    ],
    outputs=[
        helper.make_tensor_value_info(name='sum_2', shape=[66, 66],
                                      elem_type=onnx.TensorProto.FLOAT),
    ]
)

onnx.checker.check_graph(graph_def_1)
print("Graph1 checks out!")
onnx.checker.check_graph(graph_def_2)
print("Graph2 checks out!")

model_def_1 = helper.make_model(graph_def_1)
onnx.checker.check_model(model_def_1)
print('model_def_1 checks out')
model_def_2 = helper.make_model(graph_def_2)
onnx.checker.check_model(model_def_2)
print('model_def_2 checks out')

onnx.save(model_def_1, "model_1.onnx")
onnx.save(model_def_2, "model_2.onnx")

# variable_length_model = update_model_dims.update_inputs_outputs_dims(
#     model=model_def,
#     input_dims={'X': [-1, 3, 224, 224],
#                 'W1': [7, 7], 'W2': [7, 7], 'W3': [7, 7], 'W4': [7, 7],
#                 'W5': [7, 7], 'scale1': [1], 'scale2': [1], 'scale3': [1],
#                 'scale4': [1], 'scale5': [1], 'bias1': [1], 'bias2': [1],
#                 'bias3': [1], 'bias4': [1], 'bias5': [1], 'mean1': [1],
#                 'mean2': [1], 'mean3': [1], 'mean4': [1], 'mean5': [1],
#                 'var1': [1], 'var2': [1], 'var3': [1], 'var4': [1],
#                 'var5': [1],
#                 },
#     output_dims={'sum': [-1, -1]}
# )
#
# onnx.checker.check_model(variable_length_model)
# print("Var length model checks out!")