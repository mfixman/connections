import torch

SPARSE_EDGE_THRESHOLD = 65_536

def sparse_relations(edges, sources, targets):
    indices = torch.stack((edges[:, 1], edges[:, 0]))
    weights = torch.ones(len(edges), device = sources.device, dtype = sources.dtype)
    adjacency = torch.sparse_coo_tensor(
        indices,
        weights,
        (len(targets), len(sources)),
        check_invariants = True,
    ).coalesce()

    forward_counts = torch.bincount(edges[:, 1], minlength = len(targets))
    reverse_counts = torch.bincount(edges[:, 0], minlength = len(sources))
    return (
        (adjacency, forward_counts),
        (adjacency.transpose(0, 1).coalesce(), reverse_counts),
    )

def sparse_relation_mean(values, transform, relation):
    adjacency, counts = relation
    mean = torch.sparse.mm(adjacency, values) / counts.clamp(min = 1).unsqueeze(1)
    # An affine transform commutes with the mean, except for empty neighbourhoods.
    return transform(mean) * (counts > 0).unsqueeze(1)
