"""Numerical check of the patched installed scorer. Requires a spare CUDA GPU.

Run in the candidate image, never against a memory-saturated serving GPU.
"""
from types import SimpleNamespace

import numpy as np
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


device = 'cuda'
torch.manual_seed(42)
logits = torch.randn(6, 128, device=device)
mapping = torch.tensor([7, 7, 2, 9, 9, 9], dtype=torch.int32, device=device)
counts = torch.zeros(10, dtype=torch.int32, device=device)
counts[7], counts[9] = 3, 26
ids = torch.zeros(10, 128, dtype=torch.int32, device=device)
ids[7, :3] = torch.tensor([64, 65, 66], device=device)
ids[9, :26] = torch.arange(64, 90, device=device)
state = SimpleNamespace(num_token_ids=SimpleNamespace(gpu=counts),
                        token_ids=SimpleNamespace(gpu=ids))
owner = SimpleNamespace(enable_adaptive_verification=False,
                        sampler=SimpleNamespace(logprobs_mode='raw_logprobs',
                                                logprob_token_ids_state=state))
sampled = torch.tensor([[65, 66, 0], [12, 0, 0], [70, 71, 0]], device=device)
accepted = torch.tensor([2, 1, 2], dtype=torch.int32, device=device)
offsets = torch.tensor([0, 2, 3, 6], dtype=torch.int32, device=device)
reference = logits.log_softmax(-1)
flat = torch.tensor([65, 66, 12, 70, 71, 0], device=device)
for topk in (-1, 3):
    result = RejectionSampler._get_logprobs_tensors(
        owner, sampled, accepted, logits, offsets, np.array([0, 2, 3, 6]),
        topk, expanded_idx_mapping=mapping, max_per_req_token_ids=26)
    torch.testing.assert_close(result.logprob_token_ids[:, 0].long(), flat)
    torch.testing.assert_close(result.logprobs[:, 0], reference.gather(1, flat[:, None]).squeeze(1))
    for row in range(6):
        slot = mapping[row].item()
        n = counts[slot].item()
        wanted = ids[slot, :n].long() if n else logits[row].topk(max(0, topk)).indices
        torch.testing.assert_close(result.logprob_token_ids[row, 1:1+len(wanted)].long(), wanted)
        torch.testing.assert_close(result.logprobs[row, 1:1+len(wanted)], reference[row, wanted])
        assert torch.isneginf(result.logprobs[row, 1+len(wanted):]).all()
    assert result.cu_num_generated_tokens == [0, 2, 3, 6]
torch.cuda.synchronize()
print('GPU score values and mixed-request mapping passed; serving still requires validation')
