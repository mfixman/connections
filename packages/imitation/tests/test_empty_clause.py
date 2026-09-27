from connections.syntax.matrix import Clause, Matrix
from imitation.representation.matrix import matrix_graph


def test_empty_clause_has_a_valid_embedding_index():
    graph = matrix_graph(Matrix((Clause(()),)), (0,))
    assert graph.nodes["clause"][0][0] == 0
