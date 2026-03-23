"""Minimal backend ACT runtime tests."""

import torch

from backend.services import act_model
from backend.services.act_execution import TemporalEnsemblingPolicy


def test_reset_inference_context():
    runtime = act_model.get_act_runtime()
    runtime.reset_inference_context()
    assert runtime.execution_policy.step == 0
    assert len(runtime.execution_policy.predictions) == 0
    print("reset_inference_context 测试通过!")


def test_temporal_blending():
    runtime = act_model.get_act_runtime()
    runtime.reset_inference_context()

    first = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    second = torch.tensor([[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]])

    out1 = runtime.blend_current_action(first)
    out2 = runtime.blend_current_action(second)

    assert torch.allclose(out1[0, 0], torch.tensor([1.0, 1.0]))
    expected = torch.tensor([7.3333335, 7.3333335])
    assert torch.allclose(out2[0, 0], expected, atol=1e-5)
    assert runtime.execution_policy.step == 2
    print("temporal_blending 测试通过!")


def test_execution_policy_in_isolation():
    policy = TemporalEnsemblingPolicy(decay=0.5)
    out1 = policy.blend(torch.tensor([[[1.0, 1.0], [2.0, 2.0]]]))
    out2 = policy.blend(torch.tensor([[[5.0, 5.0], [6.0, 6.0]]]))
    assert torch.allclose(out1[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(out2[0, 0], torch.tensor([4.0, 4.0]))
    print("execution policy 单测通过!")
