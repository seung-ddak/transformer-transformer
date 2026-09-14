"""학습 데이터용 로봇이 만들어지고 토큰화되는 과정을 파일로 남긴다.

    MUJOCO_GL=egl python explore/step1_generated_robots.py

seed 0, 1, 2 마다
1. 부품 XML (assets/mjcf/wheeled_bimanual/*.xml) 을 무작위로 조립한 MJCF → explore/out/generated/seed_N.xml
2. 그 MJCF 를 토큰화한 표                                                → explore/out/generated/seed_N_tokens/*.csv
3. 로봇 사진                                                             → explore/out/generated/robots.png
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from dm_control import mjcf

from t2.env.mj_utils import render_opt, set_up_default_scene
from t2.robogen.wheeled_bimanual import wheeled_bimanual
from t2.robotok.io import serialize
from t2.robotok.tokenizer import preprocess_mjcf, tokenize

OUT = Path("explore/out/generated")
GROUPS = ["link", "dyna_joint", "fixed_joint", "actuator"]


def save_csv(robot, folder):
    folder.mkdir(parents=True, exist_ok=True)
    data = {k: np.asarray(v, dtype=np.float64) for k, v in serialize(robot).items()}
    data = {k: v.reshape(len(v), -1) for k, v in data.items()}
    for group in GROUPS:
        keys = sorted(k for k in data if k.startswith(group + "/"))
        header = []
        for k in keys:
            name, width = k[len(group) + 1 :], data[k].shape[1]
            header += [name] if width == 1 else [f"{name}[{i}]" for i in range(width)]
        table = np.concatenate([data[k] for k in keys], 1)
        np.savetxt(folder / f"{group}.csv", table, delimiter=",", header=",".join(header),
                   comments="", fmt="%.4g")


def render(mj_robot):
    scene = set_up_default_scene(mj_robot, add_plane=True, add_vis_cam=True, plane_z_pos=0.0,
                                 vis_cam_pos=(4.0, 2.6, 1.2))
    getattr(scene.visual, "global").offwidth = 640
    getattr(scene.visual, "global").offheight = 640
    physics = mjcf.Physics.from_mjcf_model(scene)
    opt = render_opt()
    opt.geomgroup[:] = 1
    return physics.render(camera_id=0, width=640, height=640, scene_option=opt)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for seed in [0, 1, 2]:
        mj_robot = wheeled_bimanual(seed)
        (OUT / f"seed_{seed}.xml").write_text(mj_robot.to_xml_string())

        mj_robot = preprocess_mjcf(mj_robot)
        robot, _, _, _ = tokenize(mj_robot=mj_robot)
        save_csv(robot, OUT / f"seed_{seed}_tokens")

        print(
            f"seed {seed}: seed_{seed}.xml → 토큰 표 "
            f"link {len(robot.links)} 행, 관절 {len(robot.dynamic_joints)} 개, 모터 {len(robot.actuators)} 개"
        )
        images.append(render(mj_robot))

    imageio.imwrite(OUT / "robots.png", np.concatenate(images, 1))
    print(f"\n저장: {OUT}/  (seed_N.xml, seed_N_tokens/*.csv, robots.png)")


if __name__ == "__main__":
    main()
