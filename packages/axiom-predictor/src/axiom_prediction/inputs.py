from .choices import GraphInputKind

from dataclasses import replace

from .representation.schema import GraphInput

GRAPH_INPUTS = list(GraphInputKind)

def select_graph_input(graph: GraphInput, kind: str) -> GraphInput:
    kind = GraphInputKind(kind)
    if kind == GraphInputKind.Full:
        return graph

    if kind == GraphInputKind.NoComplements:
        return replace(graph, edges = {**graph.edges, "complement": []})

    if kind == GraphInputKind.NoTerms:
        edges = {
            **graph.edges,
            **{name: [] for name in ("atom", "arg_term", "arg_var", "sym")},
        }

        return replace(
            graph,
            nodes = {**graph.nodes, "term": [], "var": []},
            edges = edges,
        )

    raise ValueError(f"unknown graph input: {kind!r}")
