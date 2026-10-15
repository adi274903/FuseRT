import sys
import onnx
import onnx_graphsurgeon as gs


def load(path):
    return gs.import_onnx(onnx.load(path))


def save(graph, path):
    onnx.save(gs.export_onnx(graph), path)
    onnx.checker.check_model(path)


def remove_identity(graph):
    """Delete Identity nodes by connecting their input straight to their consumers."""
    for node in list(graph.nodes):
        if node.op != "Identity":
            continue
        inp, out = node.inputs[0], node.outputs[0]
        if out in graph.outputs:
            continue
        for consumer in list(out.outputs):
            for i, t in enumerate(consumer.inputs):
                if t is out:
                    consumer.inputs[i] = inp
        node.outputs.clear()


def dead_code_elimination(graph):
    """Delete nodes whose results nothing uses."""
    graph.cleanup()
    graph.toposort()


PASSES = [remove_identity, dead_code_elimination]


def optimize(graph):
    print(f"start: {len(graph.nodes)} nodes")
    for p in PASSES:
        before = len(graph.nodes)
        p(graph)
        print(f"{p.__name__:28s}{before} -> {len(graph.nodes)}")
    return graph


if __name__ == "__main__":
    g = load(sys.argv[1])
    g = optimize(g)
    save(g, sys.argv[2])
    print("saved", sys.argv[2])