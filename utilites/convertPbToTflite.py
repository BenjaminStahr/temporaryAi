import tensorflow as tf
from tensorflow import lite

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    return (inputs, outputs)

path = r'C:\Users\Benja\Desktop\modelForKI\frozen_inference_graph.pb'
#print(analyze_inputs_outputs(load_graph(path)))
gf = tf.GraphDef()
gf.ParseFromString(open(path,'rb').read())
# it seems like you get input and output names with this line
print([n.name + '=>' +  n.op for n in gf.node if n.op in ( 'Softmax','Mul')])
print(gf.node[-1].name)
input_arrays = ['image_tensor']
output_arrays = ['raw_detection_scores']
converter = lite.TFLiteConverter.from_frozen_graph()
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)