"""Public architecture and prediction contracts."""
import pytest
import torch

from statevid import StateVid, build_model


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("task,kwargs,expected", [
    ("bcd", {}, 1601128), ("scd", {}, 1699960),
    ("scd", {"num_classes": 5}, 1699096), ("scd", {"num_classes": 6}, 1699528),
    ("bda", {}, 1674304), ("cc", {"vocab_size": 404}, 1806468),
    ("cc", {"vocab_size": 297}, 1792665),
])
def test_default_model_contract(task, kwargs, expected):
    model = build_model(task=task, **kwargs).eval()
    assert isinstance(model, StateVid)
    assert sum(p.numel() for p in model.parameters()) == expected
    a = torch.randn(2,3,32,40)
    with torch.inference_mode():
        result = model(a,a.flip(-1),torch.tensor([[1,2],[1,2]])) if task == "cc" else model(a,a.flip(-1))
    assert all(torch.isfinite(x).all() for x in result.values())
    if task == "cc":
        with torch.inference_mode():
            words = model.generate(a,a.flip(-1),1,2,max_length=8,beam_size=3)
        assert words.shape == (2,8) and torch.all(words[:,0] == 1)
        assert model.token_embedding.weight.shape == (kwargs["vocab_size"],128)
    else:
        assert all(x.shape[-2:] == a.shape[-2:] for x in result.values())


@pytest.mark.parametrize("depth,blocks,expected", [
    (3,[3,6,15],1013318), (5,[5,10,25],1601128), (7,[7,14,35],2188938),
])
def test_capacity_depth(depth, blocks, expected):
    model = build_model(task="bcd",depth_factor=depth)
    assert model.parameter_report()["stage_blocks"] == blocks
    assert model.parameter_report()["total"] == expected


def test_component_switches_remove_their_parameters():
    full = build_model(task="bcd")
    base = build_model(task="bcd",state_update=False,modulation=False)
    assert sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in base.parameters()) == 28520
    image = torch.randn(2,3,32,32)
    loss = base(image,image.flip(-1))["change_logits"].square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in base.student.parameters())
