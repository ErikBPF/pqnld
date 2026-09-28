"""CPU routing regression against an installed vLLM tree; no model/GPU needed.

python check_mtp_scores.py /usr/local/lib/python3.12/dist-packages/vllm
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import numpy as np


root = Path(sys.argv[1])
source = ast.parse((root / "v1/worker/gpu/spec_decode/rejection_sampler.py").read_text())
cls = next(node for node in source.body if isinstance(node, ast.ClassDef)
           and any(isinstance(m, ast.FunctionDef) and m.name == "_get_logprobs_tensors" for m in node.body))
method = next(m for m in cls.body if isinstance(m, ast.FunctionDef)
              and m.name == "_get_logprobs_tensors")
method.decorator_list = []
calls = []


class Flatten:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


scope = {"torch": torch, "NO_LOGPROBS": -1, "_flatten_sampled_kernel": Flatten(),
         "compute_topk_scores": lambda *args, **kwargs: calls.append((args, kwargs)) or kwargs}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), "installed-rejection-method", "exec"), scope)
state = object()
self = SimpleNamespace(enable_adaptive_verification=False,
                       sampler=SimpleNamespace(logprobs_mode="raw_logprobs", logprob_token_ids_state=state))
args = (self, torch.tensor([[65], [66]]), torch.tensor([1, 1]),
        torch.zeros(3, 100), torch.tensor([0, 2, 3]), torch.tensor([0, 2, 3]).numpy(), 0)
method_args = {arg.arg for arg in method.args.args}
assert "expanded_idx_mapping" in method_args, "speculative scorer lacks explicit-ID request mapping"
mapping = torch.tensor([7, 7, 2])
out = scope[method.name](*args, expanded_idx_mapping=mapping, max_per_req_token_ids=3)
assert out["expanded_idx_mapping"] is mapping
assert out["logprob_token_ids_state"] is state
assert out["max_per_req_token_ids"] == 3
assert calls[-1][0][3] == [0, 2, 3], "request boundaries changed"
assert not out["logits_mode"]
scope[method.name](*args[:-1], -1, expanded_idx_mapping=mapping, max_per_req_token_ids=3)
assert calls[-1][0][1] == 0, "explicit IDs must score even without ordinary logprobs"
calls.clear()
assert scope[method.name](*args[:-1], -1, expanded_idx_mapping=mapping,
                         max_per_req_token_ids=0) is None
assert not calls, "ordinary no-logprobs completion must retain its fast path"
self.enable_adaptive_verification = True
scope[method.name](*args, expanded_idx_mapping=mapping, max_per_req_token_ids=3)
assert torch.equal(calls[-1][0][3], args[4]), "adaptive boundaries must use device offsets"
self.enable_adaptive_verification = False

# Exercise the installed chunk orchestration with noncontiguous request slots.
chunk_method = next(m for m in cls.body if isinstance(m, ast.FunctionDef)
                    and m.name == "_verify_in_chunks")
chunk_iterator = next(m for m in source.body if isinstance(m, ast.FunctionDef)
                      and m.name == "_iter_request_chunks")
scope.update(np=np, PROCESSED_LOGPROBS_MODES=("processed_logprobs", "processed_logits"),
             LogprobsTensors=SimpleNamespace(cat=lambda chunks, **kw: (chunks, kw)))
module.body = [module.body[0], chunk_iterator, chunk_method]
exec(compile(ast.fix_missing_locations(module), "installed-chunk-method", "exec"), scope)
chunk_calls = []


def record_scores(*args):
    chunk_calls.append(args)
    return args


batch = SimpleNamespace(
    cu_num_logits_np=np.array([0, 2, 3, 6]), cu_num_logits=torch.tensor([0, 2, 3, 6]),
    num_reqs=3, idx_mapping=torch.tensor([7, 2, 9]), idx_mapping_np=np.array([7, 2, 9]),
    expanded_idx_mapping=torch.tensor([7, 7, 2, 9, 9, 9]),
    expanded_local_pos=torch.tensor([0, 1, 0, 0, 1, 2]))
self._verify = lambda logits, *args: (logits, torch.zeros(len(args[4]), 1, dtype=torch.long),
                                     torch.ones(len(args[4]), dtype=torch.long))
self._get_logprobs_tensors = record_scores
def max_ids(indices):
    assert list(indices) == [7, 2, 9], "all chunks need the full batch width"
    return max({7: 3, 2: 0, 9: 26}[i] for i in indices)


self.sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=max_ids)
logits = torch.arange(600).reshape(6, 100).float()
_, _, gathered = scope[chunk_method.name](self, logits, batch, None,
                                        torch.zeros(6, dtype=torch.long), torch.arange(6), 3, -1)
assert len(chunk_calls) == 2
assert chunk_calls[0][6].tolist() == [7, 7, 2]
assert chunk_calls[1][6].tolist() == [9, 9, 9]
assert all(call[7] == 26 for call in chunk_calls), "chunk widths must share batch maximum"
assert chunk_calls[0][4].tolist() == [0, 2, 3]
assert chunk_calls[1][4].tolist() == [0, 3]
assert torch.equal(chunk_calls[1][2], logits[3:])
assert gathered[1]["cu_num_generated_tokens"] == [0, 2, 3, 6]
print("MTP explicit-ID routing checks passed (CPU only; not kernel/serving validation)")
