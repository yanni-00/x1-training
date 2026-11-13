# SPDX-License-Identifier: BSD-3-Clause
"""F1 checkpoint provenance around the unchanged DH PPO training algorithm."""
from .dh_on_policy_runner import DHOnPolicyRunner
from humanoid.f1_contract import verify_checkpoint, write_checkpoint_metadata


class F1OnPolicyRunner(DHOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        write_checkpoint_metadata(path, self.env.cfg.asset.name, self.it + 1,
                                  getattr(self, 'parent_checkpoint', None),
                                  physical_model=getattr(self.env.cfg, 'owner_physical_model', None),
                                  goal_revision=getattr(self.env.cfg.env, 'goal_revision', 'F1-WALK-v1'))

    def load(self, path, load_optimizer=True):
        path, metadata = verify_checkpoint(path)
        infos = super().load(str(path), load_optimizer=load_optimizer)
        if self.current_learning_iteration != metadata['iteration']:
            raise ValueError('Checkpoint tensor iteration differs from F1 metadata')
        self.parent_checkpoint = {'path': str(path), 'sha256': metadata['checkpoint_sha256'],
                                  'task': metadata['task'], 'iteration': metadata['iteration']}
        return infos
