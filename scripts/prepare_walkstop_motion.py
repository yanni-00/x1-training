"""[OMA] Label kinematic support candidates; never claims dynamic stability."""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--height_quantile',type=float,default=.25);p.add_argument('--speed_quantile',type=float,default=.5)
    a=p.parse_args()
    if not (0<a.height_quantile<1 and 0<a.speed_quantile<1): raise ValueError('Invalid quantile setting')
    with np.load(a.input,allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
    names=arrays['body_names'].tolist(); feet=[names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
    pos=arrays['body_pos_w'][:,feet];vel=arrays['body_lin_vel_w'][:,feet]
    heights=pos[:,:,2].max(axis=1); speed=np.linalg.norm(vel,axis=2).max(axis=1)
    height_limit=float(np.quantile(heights,a.height_quantile));speed_limit=float(np.quantile(speed,a.speed_quantile))
    mask=(heights<=height_limit)&(speed<=speed_limit)
    boundaries=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(boundaries==1);ends=np.flatnonzero(boundaries==-1)
    support=[]
    for lo,hi in zip(starts,ends):
        # Select the least-moving frame from each contiguous candidate interval.
        support.append(int(lo+np.argmin(speed[lo:hi])))
    support=np.asarray(support,dtype=np.int64)
    if len(support)<2: raise RuntimeError('Insufficient kinematic support candidates; inspect data before training')
    arrays['support_frames']=support
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,**arrays)
    report={'source':a.input,'output':a.output,'candidate_count':len(support),'height_quantile':a.height_quantile,'speed_quantile':a.speed_quantile,'height_threshold_m':height_limit,'speed_threshold_mps':speed_limit,'support_frames':support.tolist(),'max_candidate_gap_frames':int(np.diff(support).max()),'kinematic_candidates_only':True,'dynamic_stability_verified':False}
    Path(a.output).with_suffix('.support.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__': main()
