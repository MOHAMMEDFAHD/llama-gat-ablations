"""Integration tests: graph classification and link prediction train end-to-end."""
import torch
import torch.nn.functional as F

from src.graph.model import GraphLlama
from src.graph.sparse_ops import batch_graphs, to_undirected
from src.utils.metrics import roc_auc
from src.utils.seed import set_seed


def cycle(n):
    return to_undirected(torch.stack([torch.arange(n), (torch.arange(n) + 1) % n]), n)


def path(n):
    return to_undirected(torch.stack([torch.arange(n - 1), torch.arange(1, n)]), n)


def test_graph_classification_cycle_vs_path_with_rwse():
    """Structure-only task (constant features): cycles vs paths. Needs structural info."""
    from src.data.structural import rwse
    set_seed(0)
    gs = []
    for n in range(5, 15):
        for f, lab in ((cycle, 0), (path, 1)):
            ei = f(n)
            gs.append({"x": torch.ones(n, 1), "edge_index": ei, "y": torch.tensor([lab]), "rw": rwse(ei, n, 8)})
    b = batch_graphs(gs); struct = {"rwse": torch.cat([g["rw"] for g in gs])}
    m = GraphLlama(1, 2, dim=32, num_heads=4, task="graph", pe="rwse", rwse_dim=8, dropout=0.0, input_dropout=0.0, attn_dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    for _ in range(150):
        loss = F.cross_entropy(m(b["x"], b["edge_index"], struct=struct, batch=b["batch"], num_graphs=b["num_graphs"]), b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (m(b["x"], b["edge_index"], struct=struct, batch=b["batch"], num_graphs=b["num_graphs"]).argmax(-1) == b["y"]).float().mean()
    assert acc == 1.0


def test_link_prediction_trains():
    set_seed(0)
    n = 60
    # two dense communities; positive edges within, negatives across
    g = torch.Generator().manual_seed(0)
    comm = torch.arange(n) % 2
    x = F.one_hot(comm, 2).float() + 0.3 * torch.randn(n, 2, generator=g)
    pairs = torch.combinations(torch.arange(n), 2).T
    within = comm[pairs[0]] == comm[pairs[1]]
    pos = pairs[:, within][:, torch.randperm(int(within.sum()), generator=g)[:200]]
    neg = pairs[:, ~within][:, torch.randperm(int((~within).sum()), generator=g)[:200]]
    tr_pos, te_pos, tr_neg, te_neg = pos[:, :150], pos[:, 150:], neg[:, :150], neg[:, 150:]
    ei = to_undirected(tr_pos, n)  # message passing only over training edges (no leakage)
    m = GraphLlama(2, 0, dim=16, num_heads=4, task="link", dropout=0.0, input_dropout=0.0, attn_dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(100):
        h = m(x, ei)
        lo = torch.cat([m.link_logits(h, tr_pos), m.link_logits(h, tr_neg)])
        y = torch.cat([torch.ones(150), torch.zeros(150)])
        loss = F.binary_cross_entropy_with_logits(lo, y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = m(x, ei)
        auc = roc_auc(torch.cat([m.link_logits(h, te_pos), m.link_logits(h, te_neg)]), torch.cat([torch.ones(50), torch.zeros(50)]))
    assert auc > 0.9, auc


def test_metrics_sanity():
    from src.utils.metrics import average_precision, macro_f1
    s = torch.tensor([0.9, 0.8, 0.3, 0.1]); l = torch.tensor([1, 1, 0, 0])
    assert roc_auc(s, l) == 1.0 and average_precision(s, l) == 1.0
    logits = torch.tensor([[2.0, 0], [0, 2.0], [2.0, 0]]); y = torch.tensor([0, 1, 1])
    assert abs(macro_f1(logits, y, 2) - ((2 / 3 + 2 / 3) / 2)) < 1e-9
