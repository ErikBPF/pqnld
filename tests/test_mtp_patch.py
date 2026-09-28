import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / 'build_examples/2x-rtx5060ti/prepare_mtp_patch.py'
spec = importlib.util.spec_from_file_location('mtp_patch', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PatchTests(unittest.TestCase):
    def test_model_runner_includes_explicit_ids(self):
        source = ('                    # Rejection sampler does not return logprob token ids.\n'
                  '                    include_token_ids=(\n'
                  '                        global_input_batch.num_draft_tokens == 0\n'
                  '                        or self.rejection_sampler is None\n'
                  '                    ),\n')
        self.assertEqual(module.patched('v1/worker/gpu/model_runner.py', source),
                         '                    include_token_ids=True,\n')

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported target'):
            module.patched('unrelated.py', '')

    def test_patch_is_validated_before_output(self):
        with patch.object(module.Path, 'read_text', return_value='old'), \
             patch.object(module, 'patched', side_effect=['new', ValueError('drift')]), \
             patch('sys.stdout') as stdout:
            with self.assertRaisesRegex(ValueError, 'drift'):
                module.generate(Path('/unused'))
            stdout.write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
