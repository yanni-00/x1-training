"""[OMA] Reproducible GM entry point: assets -> FK -> support labels -> native PPO."""
import argparse, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, tarfile, urllib.request
from urllib.parse import unquote, urlparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/whole_body_tracking'
ASSETS=SOURCE/'whole_body_tracking/assets'
ASSET_URL='https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz'
ASSET_SHA='b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674'
CSV_SHA='51cf310f14ecbb8be1f41ea09345df0c89024ff884fb9ab881cac339f820c787'


def main():
    p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=256);p.add_argument('--iterations',type=int,default=64);p.add_argument('--seed',type=int,default=1);p.add_argument('--run_name',default='oma_walkstop_smoke');p.add_argument('--save_interval',type=int,default=16);p.add_argument('--headless',action='store_true')
    # Same signed-URL + checksum route as the established G1 train.py.
    p.add_argument('--gm_resume_url');p.add_argument('--checkpoint_sha256')
    p.add_argument('--native_crosscheck', action='store_true',
                   help='Evaluate the supplied fixed checkpoint; no PPO updates or environment installation')
    args,extra=p.parse_known_args()
    if args.native_crosscheck and not args.gm_resume_url:
        raise ValueError('Native crosscheck requires the already authorized source checkpoint')
    if min(args.num_envs,args.iterations,args.save_interval)<=0:raise ValueError('Training sizes must be positive')
    if bool(args.gm_resume_url)!=bool(args.checkpoint_sha256):raise ValueError('Resume URL and checkpoint checksum must be supplied together')
    if args.checkpoint_sha256 and not re.fullmatch(r'[0-9a-f]{64}',args.checkpoint_sha256):raise ValueError('Invalid checkpoint checksum')
    os.chdir(ROOT)
    os.environ['OMA_CHECKPOINT_RELAY_DIR']=str(ROOT)
    os.environ['PYTHONPATH']=str(SOURCE)+os.pathsep+os.environ.get('PYTHONPATH','')
    # Diagnostic mode uses only the image's existing Lab/Torch/Numpy stack.
    # It does not import the PPO exporter/WandB path or install dependencies.
    if not args.native_crosscheck:
        missing=[pkg for mod,pkg in [('psutil','psutil'),('onnxscript','onnxscript'),('wandb','wandb>=0.19'),('toml','toml')] if importlib.util.find_spec(mod) is None]
        if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
    if args.native_crosscheck and not (ASSETS/'unitree_description/urdf/g1/main.urdf').is_file():
        raise RuntimeError('Diagnostic mode requires the assets already frozen in this repository')
    if not (ASSETS/'unitree_description/urdf/g1/main.urdf').is_file():
        archive=ROOT/'data/unitree_description.tar.gz';archive.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(ASSET_URL,timeout=120) as src,archive.open('wb') as dst:
            shutil.copyfileobj(src,dst)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=ASSET_SHA:raise RuntimeError('Robot asset archive changed; freeze and review new version')
        ASSETS.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,'r:gz') as tar:
            for m in tar.getmembers():
                if not (ASSETS/m.name).resolve().is_relative_to(ASSETS.resolve()) or m.issym() or m.islnk():raise RuntimeError('Unsafe asset archive')
            tar.extractall(ASSETS)
    # Freeze the same public G1 assets in Git, as in the existing G1 pipeline.
    manifest=json.loads((ROOT/'configs/g1_assets.json').read_text())
    for item in manifest['files']:
        asset=ASSETS/'unitree_description'/item['path']
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Frozen robot asset missing or changed: '+item['path'])
    print('[OMA] Frozen G1 robot assets verified; no live asset fetch required',flush=True)
    csv=ROOT/'data/raw/walk2_subject1.csv'
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=CSV_SHA:raise RuntimeError('Starter motion checksum mismatch')
    generated=ROOT/'data/generated';generated.mkdir(parents=True,exist_ok=True)
    converted=generated/'walk2_subject1.npz';labeled=generated/'walk2_subject1_walkstop.npz'
    print('[OMA] Stage: exact-articulation FK reference conversion',flush=True)
    cmd=[sys.executable,str(ROOT/'scripts/csv_to_npz.py'),'--input_file',str(csv),'--input_fps','30','--output_fps','50','--output_name','oma_walk2_subject1','--output_file',str(converted)]
    if args.headless:cmd.append('--headless')
    subprocess.run(cmd,check=True)
    if not converted.is_file():
        raise RuntimeError("FK worker returned without the required reference NPZ")
    import numpy as np
    with np.load(converted,allow_pickle=False) as data:
        required=["joint_pos","joint_vel","body_pos_w","body_quat_w","body_lin_vel_w","body_ang_vel_w"]
        if any(not np.isfinite(data[key]).all() for key in required):
            raise RuntimeError("Non-finite FK reference")
        if data["joint_pos"].shape[1]!=29 or data["joint_vel"].shape!=data["joint_pos"].shape:
            raise RuntimeError("FK joint dimensions are incompatible")
    print("[OMA] FK worker exited and reference arrays validated",flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_walkstop_motion.py'),'--input',str(converted),'--output',str(labeled)],check=True)
    print('[OMA] Stage: fixed-weight native evaluation' if args.native_crosscheck else '[OMA] Stage: native BeyondMimic PPO walk/hold/resume training',flush=True)
    train=ROOT/'scripts/rsl_rl/train.py'
    argv=[sys.executable,str(train),'--task','Tracking-WalkStop-G1-v0','--motion_file',str(labeled),'--num_envs',str(args.num_envs),'--max_iterations',str(args.iterations),'--seed',str(args.seed),'--run_name',args.run_name,'--experiment_name','g1_walk_stop','--logger','tensorboard']
    if args.headless:argv.append('--headless')
    if args.gm_resume_url:
        # Match the native agent's actual experiment_name (g1_flat), not the
        # unused --experiment_name CLI override. Never silently start fresh.
        match=re.search(r'model_(\d+)_',unquote(urlparse(args.gm_resume_url).path))
        if not match:raise ValueError('Resume URL lacks a checkpoint number')
        resume_dir=ROOT/'logs/rsl_rl/g1_flat/gm_resume';resume_dir.mkdir(parents=True,exist_ok=True)
        # The existing G1 pipeline documents that GM binds its first PT
        # directory. Resume downloads appear here before any new training save.
        os.environ['OMA_CHECKPOINT_RELAY_DIR']=str(resume_dir)
        os.environ['OMA_CHECKPOINT_UPLOAD_WAIT_S']='120'
        name=f'model_{match.group(1)}.pt';checkpoint_path=resume_dir/name
        print('[OMA] Downloading GM resume checkpoint '+name,flush=True)
        with urllib.request.urlopen(args.gm_resume_url,timeout=120) as response,checkpoint_path.open('wb') as destination:
            shutil.copyfileobj(response,destination)
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=args.checkpoint_sha256:
            raise RuntimeError('Resume checkpoint checksum mismatch')
        import torch
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        runtime=checkpoint['infos']['oma_runtime']
        if runtime['reference_sha256']!=hashlib.sha256(labeled.read_bytes()).hexdigest():
            raise RuntimeError('Resume reference differs from the exact trained motion')
        if not runtime['actor_obs_normalization']:
            raise RuntimeError('Resume model lacks the required native observation normalization')
        print('[OMA] Verified resume checkpoint: '+json.dumps({'file':name,'sha256':args.checkpoint_sha256,'saved_iteration':checkpoint['iter'],'reference_matches':True}),flush=True)
        argv+=['--resume','True','--load_run','gm_resume','--checkpoint',name]
    if args.native_crosscheck:
        # Reuse the existing download, reference conversion and SDK-bound PT
        # directory above. This branch constructs no optimizer and installs no
        # environment. The CSV bundle is diagnostic data, not new weights.
        argv=[sys.executable,str(ROOT/'scripts/rsl_rl/eval_native_crosscheck.py'),
              '--checkpoint_file',str(checkpoint_path),'--motion_file',str(labeled),
              '--protocol_manifest',str(ROOT/'configs/native_eval_protocols/manifest.json'),
              '--output_dir',str(ROOT/'logs/native_crosscheck'),
              '--bundle_file',str(resume_dir/'model_isaac_csv.pt'),
              '--checkpoint_sha256',args.checkpoint_sha256]
        if args.headless:argv.append('--headless')
        if extra:raise ValueError('Native crosscheck does not accept training overrides')
        print('[OMA] Stage: fixed-weight native diagnostic; zero PPO updates',flush=True)
        os.execv(sys.executable,argv)
    argv+=extra+[f'agent.save_interval={args.save_interval}']
    # exec keeps the PID monitored by gm-run; a detached grandchild loses SDK model tracking.
    os.execv(sys.executable,argv)

if __name__=='__main__':main()
