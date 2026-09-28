"""Print a reviewable patch for the inspected vLLM snapshot; never edit the engine."""
import difflib
from pathlib import Path
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("engine source differs from inspected snapshot: " + old)
    return text.replace(old, new, 1)


def patched(relative, text):
    if relative.endswith("rejection_sampler.py"):
        text = replace_once(text,
            "        max_num_logprobs: int,\n    ) -> LogprobsTensors | None:\n        if max_num_logprobs == NO_LOGPROBS:",
            "        max_num_logprobs: int,\n        expanded_idx_mapping: torch.Tensor,\n        max_per_req_token_ids: int,\n    ) -> LogprobsTensors | None:\n        if max_num_logprobs == NO_LOGPROBS and max_per_req_token_ids == 0:")
        text = replace_once(text, "            max_num_logprobs,\n            flat_sampled,",
            "            max(0, max_num_logprobs),\n            flat_sampled,")
        text = replace_once(text, "            cu_num_generated_tokens,\n            logits_mode=",
            "            cu_num_generated_tokens,\n            logprob_token_ids_state=self.sampler.logprob_token_ids_state,\n            expanded_idx_mapping=expanded_idx_mapping,\n            max_per_req_token_ids=max_per_req_token_ids,\n            logits_mode=")
        text = replace_once(text, "                max_num_logprobs,\n            )",
            "                max_num_logprobs,\n                input_batch.expanded_idx_mapping[lo:hi],\n                self.sampler.logprob_token_ids_state.max_num_token_ids(\n                    input_batch.idx_mapping_np\n                ),\n            )")
    elif relative == "v1/worker/gpu/model_runner.py":
        text = replace_once(text,
            "                    # Rejection sampler does not return logprob token ids.\n                    include_token_ids=(\n                        global_input_batch.num_draft_tokens == 0\n                        or self.rejection_sampler is None\n                    ),\n", "                    include_token_ids=True,\n")
    else:
        raise ValueError("unsupported target: " + relative)
    return text


def generate(root):
    diffs = []
    for relative in ("v1/worker/gpu/spec_decode/rejection_sampler.py", "v1/worker/gpu/model_runner.py"):
        old = (root / relative).read_text()
        new = patched(relative, old)
        diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile="a/vllm/" + relative, tofile="b/vllm/" + relative))
    return ''.join(diffs)


if __name__ == "__main__":
    sys.stdout.write(generate(Path(sys.argv[1])))
