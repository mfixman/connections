from dataclasses import replace

from .representation.schema import GraphInput

GRAPH_INPUTS = ("full", "no-complements", "no-terms")


def select_graph_input(graph: GraphInput, kind: str) -> GraphInput:
    if kind == "full":
        return graph
    if kind == "no-complements":
        return replace(graph, edges={**graph.edges, "complement": []})
    if kind == "no-terms":
        edges = {**graph.edges, **{name: [] for name in ("atom", "arg_term", "arg_var", "sym")}}
        return replace(graph, nodes={**graph.nodes, "term": [], "var": []}, edges=edges)
    raise ValueError(f"unknown graph input: {kind!r}")
