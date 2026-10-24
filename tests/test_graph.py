import math

import pytest
import torch

from src.data.structural import normalized_laplacian_eigs, rwse
from src.graph.gat import GATLayer
from src.graph.graph_block import GraphLlamaBlock
from src.graph.graph_rope import EdgeRotary, rotate_pairs
from src.graph.grouped_attention import GraphGQAttention
from src.graph.model import GraphLlama
from src.graph.sparse_ops import apply_self_loop_policy, batch_graphs, coalesce, segment_softmax, to_undirected
from src.training.graph_trainer import train_node_classifier
from src.utils.seed import set_seed



@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def random_graph(n=12, p=0.3, seed=0, isolated=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g) < p).triu(1)
    a[:, n - isolated:] = False; a[n - isolated:, :] = False
    ei = a.nonzero().T
    return to_undirected(ei, n)


def perm_graph(x, ei, perm):
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm))
    return x[perm], inv[ei]  # node new index k holds old node perm[k]


# ---------------- sparse primitives ----------------
def test_segment_softmax_matches_pyg_and_is_stable():
    from torch_geometric.utils import softmax as pyg_softmax
    idx = torch.tensor([0, 0, 1, 1, 1, 3])
    s = torch.randn(6, 4) * 50 + 1e4  # large logits
    ours = segment_softmax(s, idx, 5)
    assert torch.allclose(ours, pyg_softmax(s, idx, num_nodes=5), atol=1e-10)
    sums = torch.zeros(5, 4).index_add_(0, idx, ours)
    assert torch.allclose(sums[[0, 1, 3]], torch.ones(3, 4)) and torch.all(sums[[2, 4]] == 0)
    assert torch.isfinite(ours).all()


def test_self_loop_policies_and_duplicates():
    ei = torch.tensor([[0, 0, 1, 1, 2], [1, 1, 0, 1, 0]])  # duplicate (0,1), existing loop (1,1)
    assert coalesce(ei, 3).shape[1] == 4
    add = apply_self_loop_policy(ei, 4, "add")
    assert ((add[0] == add[1]).sum() == 4)  # exactly one loop per node, incl. isolated node 3
    rem = apply_self_loop_policy(ei, 4, "remove")
    assert (rem[0] == rem[1]).sum() == 0


def test_isolated_node_no_nan_and_zero_message():
    attn = GraphGQAttention(16, 4, 2)
    x = torch.randn(5, 16)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])  # nodes 3,4 have no in-edges
    out = attn(x, ei)
    assert torch.isfinite(out).all() and torch.allclose(out[3:], torch.zeros(2, 16))
    out2 = attn(x, apply_self_loop_policy(ei, 5, "add"))
    assert out2[3].abs().sum() > 0  # with self loops, isolated nodes attend to themselves


def test_directed_edges_respected():
    attn = GraphGQAttention(8, 2)
    x = torch.randn(3, 8)
    ei = torch.tensor([[0], [1]])  # only 0 -> 1
    out = attn(x, ei)
    x2 = x.clone(); x2[1] += 5.0  # changing the target must not affect the source's output
    assert torch.allclose(attn(x2, ei)[0], out[0])


# ---------------- GAT correctness ----------------
def test_manual_gat_matches_pyg_gatconv():
    from torch_geometric.nn import GATConv
    set_seed(0)
    n, din, dout, H = 10, 7, 5, 3
    ei = random_graph(n, 0.3)
    ours = GATLayer(din, dout, H, concat=True).eval()
    ref = GATConv(din, dout, heads=H, concat=True, add_self_loops=True).double().eval()
    with torch.no_grad():
        ref.lin.weight.copy_(ours.lin.weight); ref.att_src.copy_(ours.att_src); ref.att_dst.copy_(ours.att_dst)
        ref.bias.copy_(ours.bias)
    x = torch.randn(n, din)
    assert torch.allclose(ours(x, apply_self_loop_policy(ei, n, "add")), ref(x, ei), atol=1e-10)


# ---------------- attention / GQA ----------------
def dense_reference(attn, x, ei):
    """Dense masked attention: explicit N x N computation for a small graph."""
    n = x.shape[0]
    q = attn.q_proj(x).view(n, attn.h, attn.d); k = attn.k_proj(x).view(n, attn.hkv, attn.d); v = attn.v_proj(x).view(n, attn.hkv, attn.d)
    k = k.repeat_interleave(attn.g, 1); v = v.repeat_interleave(attn.g, 1)
    mask = torch.zeros(n, n, dtype=torch.bool); mask[ei[1], ei[0]] = True  # row i attends to col j
    s = torch.einsum("ihd,jhd->hij", q, k) / math.sqrt(attn.d)
    s = s.masked_fill(~mask, float("-inf"))
    w = torch.nan_to_num(torch.softmax(s, -1))
    return attn.o_proj(torch.einsum("hij,jhd->ihd", w, v).reshape(n, -1))


@pytest.mark.parametrize("hkv", [8, 2, 1])
def test_sparse_attention_matches_dense(hkv):
    set_seed(0)
    attn = GraphGQAttention(32, 8, hkv)
    ei = apply_self_loop_policy(random_graph(9), 9, "add")
    x = torch.randn(9, 32)
    assert torch.allclose(attn(x, ei), dense_reference(attn, x, ei), atol=1e-10)


def test_graph_gqa_equals_mha_with_duplicated_kv():
    set_seed(0)
    g, m = GraphGQAttention(32, 8, 2), GraphGQAttention(32, 8, 8)
    d = g.d
    with torch.no_grad():
        m.q_proj.weight.copy_(g.q_proj.weight); m.o_proj.weight.copy_(g.o_proj.weight)
        for nm in ("k_proj", "v_proj"):
            getattr(m, nm).weight.copy_(getattr(g, nm).weight.view(2, d, 32).repeat_interleave(4, 0).reshape(8 * d, 32))
    ei = apply_self_loop_policy(random_graph(10), 10, "add"); x = torch.randn(10, 32)
    assert torch.allclose(g(x, ei), m(x, ei), atol=1e-10)
    assert g.k_proj.weight.numel() * 4 == m.k_proj.weight.numel()


# ---------------- rotary properties ----------------
def test_node_rotary_depends_only_on_phase_difference():
    q = torch.randn(1, 1, 8); k = torch.randn(1, 1, 8)
    pi, pj = torch.randn(1, 1, 4), torch.randn(1, 1, 4); c = torch.randn(1, 1, 4)
    s1 = (rotate_pairs(q, pi) * rotate_pairs(k, pj)).sum(); s2 = (rotate_pairs(q, pi + c) * rotate_pairs(k, pj + c)).sum()
    assert torch.allclose(s1, s2)


def test_edge_rotary_antisymmetric_and_zero_on_loops():
    er = EdgeRotary(6, 2, 8)
    h = torch.randn(4, 6)
    ei = torch.tensor([[0, 1, 2], [1, 0, 2]])
    ph = er.phases(h, ei)
    assert torch.allclose(ph[0], -ph[1]) and torch.allclose(ph[2], torch.zeros_like(ph[2]))


# ---------------- permutation equivariance ----------------
def make_struct(ei, n):
    sp = normalized_laplacian_eigs(ei, n, 6)
    return {"lap_vecs": sp.vecs.double(), "lap_vals": sp.vals.double(), "rwse": rwse(ei, n, 8).double()}


@pytest.mark.parametrize("arm", [dict(rotary=None), dict(rotary="rwse"), dict(rotary="edge"),
                                 dict(rotary="rwse", struct_bias=True), dict(rotary="spectral"),
                                 dict(pe="rwse"), dict(rotary="rwse", num_kv_heads=2)])
def test_model_permutation_equivariance(arm):
    set_seed(0)
    n = 14
    ei = random_graph(n, 0.25, seed=3, isolated=2)
    x = torch.randn(n, 5)
    st = make_struct(ei, n)
    m = GraphLlama(5, 3, dim=16, num_heads=4, lap_k=6, rwse_dim=8, **arm).eval()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    xp, eip = perm_graph(x, ei, perm)
    # structural inputs transformed consistently (for LapPE this assumes the SAME eigenbasis - see below)
    stp = {k: (v[perm] if v.shape[0] == n else v) for k, v in st.items()}
    assert torch.allclose(m(xp, eip, struct=stp), m(x, ei, struct=st)[perm], atol=1e-9)


def test_index_rope_breaks_equivariance():
    set_seed(0); n = 10
    ei = random_graph(n, 0.4); x = torch.randn(n, 5)
    m = GraphLlama(5, 3, dim=16, num_heads=4, rotary="index").eval()
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(2))
    xp, eip = perm_graph(x, ei, perm)
    assert not torch.allclose(m(xp, eip), m(x, ei)[perm], atol=1e-6)


def test_rwse_recomputed_is_equivariant():
    n = 15; ei = random_graph(n, 0.3, seed=5, isolated=1)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    _, eip = perm_graph(torch.zeros(n, 1), ei, perm)
    assert torch.allclose(rwse(eip, n, 8), rwse(ei, n, 8)[perm], atol=1e-6)
    assert torch.allclose(rwse(ei, n, 8, "eig"), rwse(ei, n, 8, "power"), atol=1e-6)


def test_lappe_basis_ambiguity_on_symmetric_graph():
    """Cycle C_8: eigenvalues 1 - cos(2 pi k / 8) are doubly degenerate. Eigenspace projectors are
    equivariant; individual recomputed eigenvectors need not be (documented limitation)."""
    n = 8
    ei = to_undirected(torch.stack([torch.arange(n), (torch.arange(n) + 1) % n]), n)
    sp = normalized_laplacian_eigs(ei, n, 2, skip_null=True)
    assert sp.selected_multiplicities == [2, 2]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    _, eip = perm_graph(torch.zeros(n, 1), ei, perm)
    spp = normalized_laplacian_eigs(eip, n, 2)
    P1 = sp.vecs.double() @ sp.vecs.double().T; P2 = spp.vecs.double() @ spp.vecs.double().T
    assert torch.allclose(P2, P1[perm][:, perm], atol=1e-5)  # projector is equivariant


# ---------------- block / model mechanics ----------------
def test_residual_identity_when_branches_zeroed():
    blk = GraphLlamaBlock(16, 4, 2).eval()
    with torch.no_grad():
        blk.attn.o_proj.weight.zero_(); blk.ffn.down_proj.weight.zero_()
    x = torch.randn(6, 16); ei = apply_self_loop_policy(random_graph(6), 6, "add")
    assert torch.allclose(blk(x, ei), x)


def test_rotary_params_receive_gradients():
    n = 10; ei = random_graph(n); st = make_struct(ei, n)
    for kind in ("spectral", "rwse", "edge"):
        m = GraphLlama(4, 2, dim=16, num_heads=4, rotary=kind, lap_k=6, rwse_dim=8)
        m(torch.randn(n, 4), ei, struct=st).sum().backward()
        rot = m.blocks[0].attn.rotary
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rot.parameters()), kind


def test_batching_equals_individual_graphs():
    set_seed(0)
    gs = [{"x": torch.randn(n, 4), "edge_index": random_graph(n, 0.4, seed=n), "y": torch.tensor([n % 2])} for n in (5, 7, 6)]
    b = batch_graphs(gs)
    m = GraphLlama(4, 2, dim=16, num_heads=4, task="graph").eval()
    out = m(b["x"], b["edge_index"], batch=b["batch"], num_graphs=3)
    for i, g in enumerate(gs):
        assert torch.allclose(out[i], m(g["x"], g["edge_index"])[0], atol=1e-10)


def test_overfit_tiny_graph():
    set_seed(0)
    n = 30; ei = random_graph(n, 0.2)
    data = {"x": torch.randn(n, 8), "y": torch.randint(0, 3, (n,)), "edge_index": ei, "num_classes": 3,
            "train_mask": torch.ones(n, dtype=torch.bool), "val_mask": torch.ones(n, dtype=torch.bool),
            "test_mask": torch.ones(n, dtype=torch.bool)}
    m = GraphLlama(8, 3, dim=32, num_heads=4, dropout=0.0, input_dropout=0.0, attn_dropout=0.0)
    r = train_node_classifier(m, data, lr=1e-2, weight_decay=0.0, epochs=200, patience=200)
    assert r["train_acc"] == 1.0


def test_trainer_resume_is_bitwise_identical(tmp_path):
    n = 25; ei = random_graph(n, 0.2)
    g = torch.Generator().manual_seed(0)
    data = {"x": torch.randn(n, 6, generator=g), "y": torch.randint(0, 2, (n,), generator=g), "edge_index": ei, "num_classes": 2,
            "train_mask": torch.arange(n) < 15, "val_mask": (torch.arange(n) >= 15) & (torch.arange(n) < 20), "test_mask": torch.arange(n) >= 20}
    set_seed(0); m1 = GraphLlama(6, 2, dim=16, num_heads=4)
    r1 = train_node_classifier(m1, data, epochs=20, patience=100)
    set_seed(0); m2 = GraphLlama(6, 2, dim=16, num_heads=4)
    ck = str(tmp_path / "c.pt")
    p = train_node_classifier(m2, data, epochs=20, patience=100, checkpoint_path=ck, checkpoint_every=10, max_epochs_this_call=10)
    assert p["finished"] is False
    set_seed(123); m3 = GraphLlama(6, 2, dim=16, num_heads=4)  # different init: must be overwritten by resume
    r3 = train_node_classifier(m3, data, epochs=20, patience=100, checkpoint_path=ck, resume=True)
    assert [h["train_loss"] for h in r1["history"]] == [h["train_loss"] for h in r3["history"]]


def test_serialization_roundtrip(tmp_path):
    n = 10; ei = random_graph(n); x = torch.randn(n, 4)
    m = GraphLlama(4, 2, dim=16, num_heads=4, rotary="edge").eval()
    torch.save(m.state_dict(), tmp_path / "m.pt")
    m2 = GraphLlama(4, 2, dim=16, num_heads=4, rotary="edge").eval(); m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    assert torch.allclose(m(x, ei), m2(x, ei))


def test_bf16_autocast_graph():
    torch.set_default_dtype(torch.float32)
    try:
        n = 20; ei = random_graph(n); x = torch.randn(n, 4)
        m = GraphLlama(4, 2, dim=16, num_heads=4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(x, ei)
        assert torch.isfinite(out.float()).all()
    finally:
        torch.set_default_dtype(torch.float64)
