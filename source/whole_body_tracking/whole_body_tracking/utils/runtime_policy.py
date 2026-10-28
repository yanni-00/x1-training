"""[OMA] RSL-RL checkpoint actor for the existing server MuJoCo play path."""
import re
import torch
from torch import nn


class RuntimeActor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        state = checkpoint["model_state_dict"]
        runtime = checkpoint["infos"]["oma_runtime"]
        if runtime["format"] != "beyondmimic_g1_runtime_v1":
            raise ValueError("Unsupported runtime model schema")
        ids = sorted(int(k.split(".")[1]) for k in state if re.fullmatch(r"actor\.\d+\.weight", k))
        layers = []
        for index, number in enumerate(ids):
            w, b = state[f"actor.{number}.weight"], state[f"actor.{number}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w); layer.bias.data.copy_(b)
            layers.append(layer)
            if index + 1 < len(ids): layers.append(nn.ELU())
        if not layers or layers[0].in_features != 160 or layers[-1].out_features != 29:
            raise ValueError("Expected BeyondMimic 160-to-29 actor")
        self.actor = nn.Sequential(*layers)
        self.normalized = runtime["actor_obs_normalization"]
        self.eps = runtime["normalizer_epsilon"]
        if self.normalized:
            self.register_buffer("mean", state["actor_obs_normalizer._mean"].clone())
            self.register_buffer("std", state["actor_obs_normalizer._std"].clone())
            if self.eps is None or self.eps <= 0 or self.mean.numel() != 160 or self.std.numel() != 160:
                raise ValueError("Invalid model observation normalizer")
        if any(not torch.isfinite(v).all() for v in self.state_dict().values()):
            raise ValueError("Non-finite runtime actor state")
        if self.normalized and (self.std < 0).any(): raise ValueError("Negative normalization standard deviation")

    def forward(self, obs):
        if self.normalized: obs = (obs - self.mean) / (self.std + self.eps)
        return self.actor(obs)


def verify_runtime_checkpoint(policy, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    actor = RuntimeActor(checkpoint).eval()
    generator = torch.Generator(device="cpu").manual_seed(17)
    observations = torch.randn((64,160), generator=generator)
    if actor.normalized: observations = observations * actor.std + actor.mean
    device = next(policy.parameters()).device
    with torch.inference_mode():
        expected = policy.actor(policy.actor_obs_normalizer(observations.to(device))).cpu()
        actual = actor(observations)
    # Same parity tolerances as the existing BeyondMimic package exporter.
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    return {"samples":64, "max_abs_error":float((actual-expected).abs().max()),
            "actor_obs_normalization":actor.normalized, "normalizer_epsilon":actor.eps}
