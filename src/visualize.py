import graphviz
from graphviz import Digraph

testconf = {'0': [2, []], '1': [2, [1]], '2': [3, [1, 1]], '3': [0, [0, 1, 0]], '4': [3, [1, 1, 1, 0]], '5': [3, [1, 1, 0, 0, 0]], '6': [3, [1, 0, 1, 0, 0, 1]], '7': [4, [1, 0, 0, 1, 1, 1, 1]], '8': [3, [0, 1, 0, 1, 0, 1, 1, 1]], '9': [0, [0, 0, 0, 0, 1, 0, 0, 0, 0]], '10': [0, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], '11': [5, [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]]}



lookup = {
    "0":"Conv_3x3",
    "1":"Conv_5x5",
    "2":"SepConv_3x3",
    "3":"SepConv_5x5",
    "4":"AvgPool_3x3",
    "5":"MaxPool_3x3"
}


def draw_conf(conf):
    child = Digraph(comment='CHILD')

    for layer_idx, layer in enumerate(conf.values()):
        print(chr(layer_idx + 97), layer[0], "L" + str(layer_idx) + ":" + lookup[str(layer[0])])
        child.node(chr(layer_idx + 97), "L" + str(layer_idx) + ":" + lookup[str(layer[0])])

    last_layer_node = chr(layer_idx + 98)
    child.node(chr(layer_idx + 98), "L" + str(layer_idx + 1) + ":" + "FullyConnected")

    edges = []
    for layer_idx, layer in enumerate(conf.values()):
        for skip_idx in range(len(layer[1])):
            if conf[skip_idx]:
                edges.append(chr(skip_idx + 97) + chr(layer_idx + 97))
        edges.append(chr(layer_idx + 97) + last_layer_node)

    child.edges(edges)
    return child

draw_conf(testconf)
