"""1 단계: RoboToken zarr 열어 보기.

    python explore/step1_inspect_zarr.py                      # mj_menagerie.zarr, UR5e
    python explore/step1_inspect_zarr.py --robot 0            # Allegro
    python explore/step1_inspect_zarr.py --path wheeled_bimanual.zarr --robot 0
"""

import argparse

import numpy as np
import zarr

MENAGERIE_NAMES = [
    "allegro", "anymal_c", "anymal_b", "h1", "go1", "go2",
    "a1", "aloha", "ur5e", "ur10e", "umi_on_legs",
]
GROUPS = ["link", "dyna_joint", "fixed_joint", "actuator"]


def split_columns(attrs: dict) -> list[tuple[str, int, int]]:
    """속성 이름 → 폭 을 (이름, 시작 칸, 끝 칸) 으로. zarr 에 알파벳 순으로 이어 붙어 있다."""
    cols, start = [], 0
    for name in sorted(attrs):
        cols.append((name, start, start + attrs[name]))
        start += attrs[name]
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="mj_menagerie.zarr")
    parser.add_argument("--robot", type=int, default=8, help="로봇 번호 (mj_menagerie 기준 8 = UR5e)")
    parser.add_argument("--group", default="link", choices=GROUPS)
    parser.add_argument("--rows", type=int, default=3, help="표시할 토큰 수")
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True, linewidth=120)

    r = zarr.open(args.path, mode="r")

    print("① 전체 구조")
    print(r.tree())

    print("\n② 토큰 한 줄의 구성 (속성: 칸 범위)")
    for g in GROUPS:
        a = r[f"hardware/{g}"]
        cols = ", ".join(f"{n}[{s}:{e}]" for n, s, e in split_columns(dict(a.attrs)))
        print(f"  {g:12s} {a.shape}  {cols}")

    print("\n③ 로봇별 행 수 (ends 차이)")
    ends = {g: r[f"hardware_meta/{g}/ends"][:].astype(int) for g in GROUPS}
    n_robots = len(ends["link"])
    names = MENAGERIE_NAMES if n_robots == len(MENAGERIE_NAMES) else [f"robot{i}" for i in range(n_robots)]
    print(f"  {'robot':12s} " + " ".join(f"{g:>11s}" for g in GROUPS))
    for i in range(n_robots):
        counts = [ends[g][i] - (ends[g][i - 1] if i > 0 else 0) for g in GROUPS]
        print(f"  {i:2d} {names[i]:9s} " + " ".join(f"{c:11d}" for c in counts))
    print("  (dyna_joint, fixed_joint 는 관절 1 개 = 2 행)")

    i, g = args.robot, args.group
    lo = ends[g][i - 1] if i > 0 else 0
    hi = ends[g][i]
    tokens = r[f"hardware/{g}"][lo:hi]
    print(f"\n④ {names[i]} 의 {g} 토큰 {hi - lo} 개 중 앞 {min(args.rows, hi - lo)} 개 (행 {lo}~{hi})")
    for t, tok in enumerate(tokens[: args.rows]):
        print(f"  토큰 {t}")
        for name, s, e in split_columns(dict(r[f"hardware/{g}"].attrs)):
            print(f"    {name:14s} {tok[s:e]}")


if __name__ == "__main__":
    main()
