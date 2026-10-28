# BeyondMimic G1-29DOF walk/stop

Independent task: Tracking-WalkStop-G1-v0. Native BeyondMimic robot, PPO, observations, rewards and terminations are retained. A support-aware reference clock controls GO/STOP, holds reference time while policy control continues, and resumes without robot resets.

This is an unvalidated implementation. CPU clock checks are independent of physical stability. The first GM run is a short infrastructure smoke, not a locomotion acceptance run.

Entry: gm-run x1-training/scripts/oma_bootstrap.py --headless --num_envs 256 --iterations 64 --seed 1

The bootstrap verifies pinned asset/data checksums, derives references with the training articulation, labels kinematic support candidates, then execs the native PPO entry point. Generated references are local to the task. Original CSV is bundled with upstream source/permit notes under data/raw.

Full state-estimation policy observations are retained to match the existing G1 deployment interface. Any new policy requires fresh export and parity checks; no robot execution is performed by this loop.

Evaluation: scripts/rsl_rl/eval_walk_stop.py produces continuous actual-state CSV and video using configs/eval_walk_stop.json. scripts/score_walk_stop.py rejects reference exhaustion during GO, malformed timebases, missing gates and unreviewed videos. The finite starter reference cannot yet provide unlimited walking; this remains an explicit later-loop requirement.

Run evaluation in the same Isaac Lab environment with this package installed (or source/whole_body_tracking on PYTHONPATH). The evaluator uses an explicitly recorded nominal protocol and aborts at the first environment termination. Initial support selection applies only to evaluation reset, never to GO/STOP transitions.

Current pilot entry: gm-run x1-training/scripts/oma_bootstrap.py --headless --num_envs 4096 --iterations 2000 --save_interval 500 --seed 1 --run_name oma_walkstop_baseline. These are preregistered experiment settings, not a performance or runtime estimate. Training resets sample GO/STOP using initial_go_probability=0.5; static evaluation initialization remains STOP.

Use the established server gmr environment for physical replay: python scripts/rsl_rl/sim2sim_walk_stop.py --checkpoint model.pt --protocol configs/eval_walk_stop.json --output_dir play. Then python scripts/score_walk_stop.py --evaluation_dir play --gates configs/behavior_gates.json. The adapter reuses the G1 MuJoCo clipped-PD/rendering workflow and reconstructs the full exact training reference, control parameters and observation normalizer from the checkpoint. Truth-state replay is a first check; deployment estimator parity remains separate.

The STOP stages use terminal stable-hold windows and separately report stable_stop_response_s; no user stop-response deadline is imposed. Finite protocol duration limits observation. Engineering behavior criteria are frozen in configs/behavior_gates.json before the pilot launch; they are not measured robot specifications.

Runtime actor comparisons temporarily disable TF32 and preserve training precision, module modes and CPU RNG; tolerance is unchanged. RSL-RL v3.1.0 EmpiricalNormalization.forward is already read-only, so the mode guard is defensive and is not evidence for the previous mismatch cause. First-save and final-save native/runtime parity must both pass.
