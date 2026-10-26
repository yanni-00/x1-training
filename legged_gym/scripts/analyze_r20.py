import numpy as np

# r20 (exp-20261026-001) S7 门控核算——三预注册门对照
# 输入: resources/play_diag_7800.csv + resources/waist_diag_7800.csv (服务器 play 产出)
GATES = [
    ('直腰门 |waist_pitch| <= 8 deg', 'waist_pitch_deg', 8.0),
    ('站高门 base_h >= 0.65 m', 'base_h', 0.65),
    ('速度门 vel_x >= 0.15', 'vel_x', 0.15),
]


def load(path):
    return np.genfromtxt(path, delimiter=',', names=True)


def main():
    pd = load('resources/play_diag_7800.csv')
    wd = load('resources/waist_diag_7800.csv')

    print('=== play_diag_7800 (cmd_vx=0.5 全程) ===')
    steady = pd[pd['step'] >= 200]
    print('steps=%d  vel_x(steady>4s)=%.3f (std %.3f)  vel_y=%.3f  vel_wz=%.3f' % (
        len(pd), steady['vel_x'].mean(), steady['vel_x'].std(),
        steady['vel_y'].mean(), steady['vel_wz'].mean()))
    print('base_h(steady)=%.3f m  grav_x(steady)=%+.3f' % (
        steady['base_h'].mean(), steady['grav_x'].mean()))

    print('=== waist_diag_7800 (0-2s 站立 / >4s 行走) ===')
    stand = wd[wd['step'] < 100]
    wsteady = wd[wd['step'] >= 200]
    for name, seg in (('stand', stand), ('walk', wsteady)):
        for j in (5, 6, 7):
            pass
    cols = list(wd.dtype.names)
    wcols = [c for c in cols if c.startswith('waist')]
    print('waist cols:', wcols)
    for name, seg in (('stand', stand), ('walk(steady)', wsteady)):
        if len(seg) == 0:
            print(f'{name}: no data')
            continue
        vals = [seg[c].mean() * 180.0 / np.pi for c in wcols]
        print('%-13s: %s   base_h=%.3f  vx=%+.3f' % (
            name, '  '.join(f'{c}={v:+.1f}deg' for c, v in zip(wcols, vals)),
            seg['base_h'].mean(), seg['vel_x'].mean()))

    print('=== 门控对照 ===')
    # waist 列名到 yaw/roll/pitch 的映射以 waist_diag 打印的关节序为准（waist0/1/2 按索引序）
    for label, key, thr in GATES:
        if key == 'vel_x':
            v = steady['vel_x'].mean()
            ok = v >= thr
        elif key == 'base_h':
            v = steady['base_h'].mean()
            ok = v >= thr
        else:
            # waist2=索引第3个腰关节；命名待与服务器打印的关节序核对
            c = wcols[2] if len(wcols) >= 3 else wcols[-1]
            v = wsteady[c].mean() * 180.0 / np.pi
            ok = abs(v) <= thr
        print('%-32s 值=%+.3f  门=%s  -> %s' % (label, v, thr, 'PASS' if ok else 'FAIL'))


if __name__ == '__main__':
    main()
