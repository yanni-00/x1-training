"""[OMA] Support-aware reference time; no simulator state writes or resets."""
import torch


class SupportPhaseClock:
    def __init__(self, count, fps, support_frames, device='cpu', acceleration=2.0):
        if fps <= 0 or acceleration <= 0:
            raise ValueError('fps and acceleration must be positive')
        self.fps = float(fps)
        self.acceleration = float(acceleration)
        self.support = torch.as_tensor(support_frames, dtype=torch.float32, device=device).unique(sorted=True)
        if self.support.numel() < 2 or not torch.isfinite(self.support).all():
            raise ValueError('At least two finite support candidates are required')
        self.phase = torch.zeros(count, device=device)
        self.rate = torch.zeros(count, device=device)
        self.target = torch.zeros(count, device=device)
        self.was_go = torch.zeros(count, dtype=torch.bool, device=device)
        self.exhausted = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids, phases):
        """Only called alongside a genuine environment reset."""
        idx = torch.searchsorted(self.support, phases).clamp(max=len(self.support)-1)
        self.phase[ids] = self.support[idx]
        self.rate[ids] = 0
        self.target[ids] = self.phase[ids]
        self.was_go[ids] = False
        self.exhausted[ids] = False

    def step(self, go, dt):
        if dt <= 0:
            raise ValueError('dt must be positive')
        go = go.to(dtype=torch.bool, device=self.phase.device)
        falling = self.was_go & ~go
        braking_distance = self.rate.square() * self.fps / (2*self.acceleration)
        earliest = self.phase + braking_distance + self.rate*self.fps*dt
        next_idx = torch.searchsorted(self.support, earliest).clamp(max=len(self.support)-1)
        self.target[falling] = self.support[next_idx[falling]]
        remaining = (self.target-self.phase).clamp_min(0)
        cap = torch.sqrt(2*self.acceleration*remaining/self.fps).clamp(max=1)
        go_rate = (self.rate+self.acceleration*dt).clamp(max=1)
        stop_rate = torch.minimum(self.rate, cap)
        new_rate = torch.where(go, go_rate, stop_rate)
        advance = (self.rate+new_rate)*0.5*dt*self.fps
        advance = torch.where(go, advance, torch.minimum(advance, remaining))
        self.phase += advance
        stopped = ~go & ((self.target-self.phase) <= 1e-5)
        self.phase[stopped] = self.target[stopped]
        new_rate[stopped] = 0
        # End-of-reference always holds; never wraps or teleports the robot.
        self.exhausted = self.phase >= self.support[-1]
        self.phase.clamp_(max=float(self.support[-1]))
        new_rate[self.exhausted] = 0
        self.rate = new_rate
        self.was_go.copy_(go)
        return self.phase, self.rate
