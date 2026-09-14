"""토큰 표의 숫자를 직접 바꾸면 로봇이 어떻게 바뀌는지 본다.

    MUJOCO_GL=egl python explore/step1_edit_token.py

1. UR5e 를 토큰 표로 바꿔 explore/out/ur5e_tokens/*.csv 로 저장한다 (VS Code 에서 열어 볼 수 있다).
2. 표의 숫자를 바꾼다.
3. 바꾼 표를 다시 로봇(MJCF)으로 되돌려 사진을 찍는다 → explore/out/step1_edit_token.png
"""

import copy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from dm_control import mjcf

from t2.env.mj_utils import render_opt, set_up_default_scene
from t2.robotok.io import deserialize, serialize
from t2.robotok.tokenizer import detokenize, preprocess_mjcf, tokenize

OUT = Path("explore/out")
GROUPS = ["link", "dyna_joint", "fixed_joint", "actuator"]


def to_tables(robot):
    data = {k: np.asarray(v, dtype=np.float64) for k, v in serialize(robot).items()}
    return {k: v.reshape(len(v), -1) for k, v in data.items()}


def save_csv(tables, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        keys = sorted(k for k in tables if k.startswith(group + "/"))
        header, cols = [], []
        for k in keys:
            arr = tables[k]
            name = k[len(group) + 1 :]
            header += [name] if arr.shape[1] == 1 else [f"{name}[{i}]" for i in range(arr.shape[1])]
            cols.append(arr)
        np.savetxt(folder / f"{group}.csv", np.concatenate(cols, 1), delimiter=",",
                   header=",".join(header), comments="", fmt="%.4g")


def render(tables, title):
    data = {}
    for k, v in tables.items():
        is_int = k.endswith("/id") or k.endswith("/type") or k == "link/contact_dim"
        data[k] = v.astype(np.int64) if is_int else v.astype(np.float32)
        if k.endswith("/id") or k.endswith("/type"):
            data[k] = data[k].reshape(-1)
    model, _, _, _ = detokenize(deserialize(data), gravcomp=True)
    model = set_up_default_scene(
        model, add_plane=True, add_vis_cam=True, plane_z_pos=0.0, vis_cam_pos=(1.66, 1.06, 0.55)
    )
    getattr(model.visual, "global").offwidth = 640
    getattr(model.visual, "global").offheight = 480
    physics = mjcf.Physics.from_mjcf_model(model)
    opt = render_opt()
    opt.geomgroup[:] = 1
    img = physics.render(camera_id=0, width=640, height=480, scene_option=opt)
    print(f"  {title}")
    return img


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    mj_robot = preprocess_mjcf(mjcf.from_path("assets/mjcf/universal_robots_ur5e/ur5e.xml"))
    robot, _, _, _ = tokenize(mj_robot=mj_robot)
    tables = to_tables(robot)
    save_csv(tables, OUT / "ur5e_tokens")
    print(f"토큰 표 저장: {OUT / 'ur5e_tokens'}/{{{','.join(GROUPS)}}}.csv\n")

    images = [render(tables, "A. 원본")]

    b = copy.deepcopy(tables)
    b["link/geom_size"][2, 0] *= 3  # link 2 (upper arm) 반지름 3 배
    b["link/mass"][2, 0] *= 3
    images.append(render(b, "B. link 표 2 행: geom_size 반지름 ×3, mass ×3"))

    c = copy.deepcopy(tables)
    c["fixed_joint/pos"][0, 2] = 0.9  # fixed_joint 0 (월드 ↔ base): 높이 0.9 m
    c["fixed_joint/rotmat"][0] = np.diag([1.0, -1.0, -1.0]).reshape(-1)  # x 축으로 180° → 뒤집기
    images.append(render(c, "C. fixed_joint 표 0 행: pos z = 0.9, rotmat = 뒤집기 (천장에 매달기)"))

    OUT.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(OUT / "step1_edit_token.png", np.concatenate(images, 1))
    print(f"\n사진 저장: {OUT / 'step1_edit_token.png'}  (왼쪽부터 A, B, C)")


if __name__ == "__main__":
    main()
