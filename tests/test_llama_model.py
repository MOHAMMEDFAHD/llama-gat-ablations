import torch

from src.data.synthetic_lm import progression_batch
from src.llama import LlamaConfig, LlamaForCausalLM, generate, preset, sample_next
from src.training.lm_trainer import load_checkpoint, save_checkpoint, train_progression_lm
from src.utils.seed import set_seed


def tiny(**kw):
    base = dict(vocab_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    base.update(kw)
    return LlamaConfig(**base)


def test_forward_backward_all_params_get_grads():
    set_seed(0)
    m = LlamaForCausalLM(tiny())
    ids = torch.randint(0, 64, (2, 10))
    out = m(ids, labels=ids)
    out.loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert out.logits.shape == (2, 10, 64)


def test_initial_loss_near_log_vocab():
    set_seed(0)
    m = LlamaForCausalLM(tiny())
    ids = torch.randint(0, 64, (8, 16))
    assert abs(float(m(ids, labels=ids).loss) - torch.log(torch.tensor(64.0)).item()) < 0.2


def test_causality_future_tokens_do_not_affect_past():
    set_seed(0)
    m = LlamaForCausalLM(tiny()).eval()
    a = torch.randint(0, 64, (1, 10)); b = a.clone(); b[0, 6:] = (b[0, 6:] + 1) % 64
    assert torch.allclose(m(a).logits[:, :6], m(b).logits[:, :6], atol=1e-5)


def test_left_padding_matches_unpadded():
    set_seed(0)
    m = LlamaForCausalLM(tiny()).eval()
    seq = torch.randint(1, 64, (1, 6))
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), seq], 1)
    mask = torch.cat([torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 6, dtype=torch.long)], 1)
    assert torch.allclose(m(seq).logits, m(padded, attention_mask=mask).logits[:, 3:], atol=1e-5)


def test_kv_cache_logits_match_full_forward():
    set_seed(0)
    m = LlamaForCausalLM(tiny()).eval()
    ids = torch.randint(0, 64, (2, 12))
    full = m(ids).logits
    out = m(ids[:, :5], use_cache=True); past = out.past_key_values; steps = [out.logits]
    for t in range(5, 12):
        o = m(ids[:, t : t + 1], past_key_values=past, use_cache=True); past = o.past_key_values; steps.append(o.logits)
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-5)
    # cache stores H_kv heads (GQA memory saving)
    assert past[0][0].shape[1] == 2


def test_cached_and_uncached_generation_identical():
    set_seed(0)
    m = LlamaForCausalLM(tiny())
    prompt = torch.randint(0, 64, (2, 4))
    g1 = generate(m, prompt, 10, use_cache=True); g2 = generate(m, prompt, 10, use_cache=False)
    assert torch.equal(g1, g2)
    gen = torch.Generator().manual_seed(1); s1 = generate(m, prompt, 10, temperature=0.9, top_k=10, use_cache=True, generator=gen)
    gen = torch.Generator().manual_seed(1); s2 = generate(m, prompt, 10, temperature=0.9, top_k=10, use_cache=False, generator=gen)
    assert torch.equal(s1, s2)


def test_sampling_filters():
    logits = torch.tensor([[5.0, 4.0, 0.0, -1.0]])
    for _ in range(50):
        assert int(sample_next(logits, 1.0, top_k=2)) in (0, 1)
    probs = logits.softmax(-1)[0]  # p0 ~ .72 -> top_p=.5 keeps only token 0
    for _ in range(50):
        assert int(sample_next(logits, 1.0, top_p=0.5)) == 0
    assert int(sample_next(logits, 0.0)) == 0


def test_weight_tying():
    m = LlamaForCausalLM(tiny(tie_word_embeddings=True))
    assert m.lm_head.weight.data_ptr() == m.embed_tokens.weight.data_ptr()
    untied = LlamaForCausalLM(tiny())
    assert m.num_parameters() == untied.num_parameters() - 64 * 64


def test_presets_build():
    for name in ["llama1-tiny", "llama2-tiny", "llama2-70b-style-tiny", "llama3-tiny"]:
        c = preset(name, vocab_size=32, hidden_size=32, num_hidden_layers=1)
        assert LlamaForCausalLM(c)(torch.randint(0, 32, (1, 4))).logits.shape == (1, 4, 32)


def test_bf16_autocast_forward():
    set_seed(0)
    m = LlamaForCausalLM(tiny())
    ids = torch.randint(0, 64, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(ids, labels=ids)
    assert torch.isfinite(out.loss)
    assert abs(float(out.loss) - float(m(ids, labels=ids).loss)) < 0.05


def test_training_learns_progressions_and_generates(tmp_path):
    cfg = tiny(vocab_size=32)
    res = train_progression_lm(cfg, steps=400, batch=32, seq_len=16, lr=3e-3, seed=0, val_every=100,
                               log_path=str(tmp_path / "lm.json"))
    h = res["history"]
    assert h["val_loss"][-1] < 0.25 * h["val_loss"][0], h["val_loss"]
    m = res["model"]
    seqs = progression_batch(16, 12, 32, generator=torch.Generator().manual_seed(123))
    out = generate(m, seqs[:, :3], 9)
    acc = float((out == seqs).float()[:, 3:].mean())
    assert acc > 0.9, acc
    # checkpoint round trip
    save_checkpoint(tmp_path / "ck.pt", m, res["optimizer"], 400)
    ck = load_checkpoint(tmp_path / "ck.pt")
    assert torch.equal(generate(ck["model_obj"], seqs[:, :3], 9), out)


def test_reproducible_under_seed():
    a = train_progression_lm(tiny(vocab_size=32), steps=20, seed=3, val_every=10)["history"]["train_loss"]
    b = train_progression_lm(tiny(vocab_size=32), steps=20, seed=3, val_every=10)["history"]["train_loss"]
    assert a == b


def test_resume_training_identical(tmp_path):
    """Train 10 steps, checkpoint, train 10 more == restore and train 10 more."""
    set_seed(0)
    cfg = tiny(vocab_size=32)
    m = LlamaForCausalLM(cfg); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step(model, o):
        ids = progression_batch(8, 12, 32); loss = model(ids, labels=ids).loss
        o.zero_grad(); loss.backward(); o.step(); return float(loss)

    for _ in range(10): step(m, opt)
    save_checkpoint(tmp_path / "r.pt", m, opt, 10)
    cont = [step(m, opt) for _ in range(10)]
    m2 = LlamaForCausalLM(cfg); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "r.pt", m2, opt2)
    resumed = [step(m2, opt2) for _ in range(10)]
    assert cont == resumed
