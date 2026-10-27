"""[OMA] Reproducible GM entry point: assets -> FK -> support labels -> native PPO."""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, tarfile, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source/whole_body_tracking'
ASSETS=SOURCE/'whole_body_tracking/assets'
ASSET_URL='https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz'
ASSET_SHA='b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674'
CSV_SHA='51cf310f14ecbb8be1f41ea09345df0c89024ff884fb9ab881cac339f820c787'


def main():
    p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=256);p.add_argument('--iterations',type=int,default=64);p.add_argument('--seed',type=int,default=1);p.add_argument('--run_name',default='oma_walkstop_smoke');p.add_argument('--headless',action='store_true')
    args,extra=p.parse_known_args()
    os.chdir(ROOT)
    os.environ['OMA_CHECKPOINT_RELAY_DIR']=str(ROOT)
    os.environ['PYTHONPATH']=str(SOURCE)+os.pathsep+os.environ.get('PYTHONPATH','')
    missing=[pkg for mod,pkg in [('psutil','psutil'),('onnxscript','onnxscript'),('wandb','wandb>=0.19'),('toml','toml')] if importlib.util.find_spec(mod) is None]
    if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
    if not (ASSETS/'unitree_description/urdf/g1/main.urdf').is_file():
        archive=ROOT/'data/unitree_description.tar.gz';archive.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(ASSET_URL,timeout=120) as src,archive.open('wb') as dst:
            import shutil;shutil.copyfileobj(src,dst)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=ASSET_SHA:raise RuntimeError('Robot asset archive changed; freeze and review new version')
        ASSETS.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,'r:gz') as tar:
            for m in tar.getmembers():
                if not (ASSETS/m.name).resolve().is_relative_to(ASSETS.resolve()) or m.issym() or m.islnk():raise RuntimeError('Unsafe asset archive')
            tar.extractall(ASSETS)
    csv=ROOT/'data/raw/walk2_subject1.csv'
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=CSV_SHA:raise RuntimeError('Starter motion checksum mismatch')
    generated=ROOT/'data/generated';generated.mkdir(parents=True,exist_ok=True)
    converted=generated/'walk2_subject1.npz';labeled=generated/'walk2_subject1_walkstop.npz'
    print('[OMA] Stage: exact-articulation FK reference conversion',flush=True)
    cmd=[sys.executable,str(ROOT/'scripts/csv_to_npz.py'),'--input_file',str(csv),'--input_fps','30','--output_fps','50','--output_name','oma_walk2_subject1','--output_file',str(converted)]
    if args.headless:cmd.append('--headless')
    subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_walkstop_motion.py'),'--input',str(converted),'--output',str(labeled)],check=True)
    print('[OMA] Stage: native BeyondMimic PPO walk/hold/resume training',flush=True)
    train=ROOT/'scripts/rsl_rl/train.py'
    argv=[sys.executable,str(train),'--task','Tracking-WalkStop-G1-v0','--motion_file',str(labeled),'--num_envs',str(args.num_envs),'--max_iterations',str(args.iterations),'--seed',str(args.seed),'--run_name',args.run_name,'--experiment_name','g1_walk_stop','--logger','tensorboard']
    if args.headless:argv.append('--headless')
    argv+=extra+['agent.save_interval=16']
    # exec keeps the PID monitored by gm-run; a detached grandchild loses SDK model tracking.
    os.execv(sys.executable,argv)

if __name__=='__main__':main()
