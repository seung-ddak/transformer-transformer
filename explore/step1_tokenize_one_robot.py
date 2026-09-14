"""1 단계: 로봇 1 대가 토큰이 되는 과정을 단계별로 출력.

    python explore/step1_tokenize_one_robot.py                       # UR5e
    python explore/step1_tokenize_one_robot.py --xml assets/mjcf/unitree_go2/go2.xml

실제 데이터 생성 경로 (t2/env/base_env.py reset_hardware) 와 같은 순서:
    MJCF → preprocess_mjcf → tokenize → Robot 객체 → serialize → zarr 행
"""

import argparse

import mujoco
import numpy as np
from dm_control import mjcf

from t2.robotok.io import serialize
from t2.robotok.tokenizer import preprocess_mjcf, tokenize


def count_model(model: mujoco.MjModel) -> str:
    n_collision = int(np.sum((model.geom_contype != 0) & (model.geom_conaffinity != 0)))
    return (
        f"body {model.nbody - 1}, geom {model.ngeom} (충돌용 {n_collision}), "
        f"joint {model.njnt}, actuator {model.nu}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="assets/mjcf/universal_robots_ur5e/ur5e.xml")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print(f"[1] 원본 MJCF: {args.xml}")
    mj_robot = mjcf.from_path(args.xml)
    raw = mjcf.Physics.from_mjcf_model(mj_robot).model.ptr
    print("    " + count_model(raw))

    print("\n[2] preprocess_mjcf (t2/robotok/tokenizer.py:701)")
    print("    링크마다 충돌 geom 1 개로 정리, 센서 제거, transform 정규화")
    mj_robot = preprocess_mjcf(mj_robot, remove_visual=False)
    physics = mjcf.Physics.from_mjcf_model(mj_robot)
    model = physics.model.ptr
    print("    " + count_model(model))

    print("\n[3] tokenize (t2/robotok/tokenizer.py:1210) → Robot 객체 (t2/robotok/token.py:286)")
    robot, jnt_map, geom_map, _ = tokenize(mj_robot=mj_robot)
    print(
        f"    Link {len(robot.links)}, DynamicJoint {len(robot.dynamic_joints)}, "
        f"FixedJoint {len(robot.fixed_joints)}, Actuator {len(robot.actuators)}"
    )

    link_to_body = {}
    for geom_id, link_id in geom_map.items():
        if geom_id >= 0:
            body_id = model.geom_bodyid[geom_id]
            link_to_body[link_id] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    dyna_to_joint = {
        d: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j, d in jnt_map.items()
    }

    print("\n    Link 토큰 (Link 클래스, token.py:101) ← MJCF <body>/<inertial>/<geom>")
    for link in sorted(robot.links, key=lambda x: x.idx):
        print(
            f"      link {link.idx:2d} ← body '{link_to_body.get(link.idx, '?')}': "
            f"{link.geom_type.value} size={np.array(link.size)} mass={link.mass:.3f} "
            f"diaginertia={np.array(link.diaginertia)}"
        )

    print("\n    DynamicJoint 토큰 (token.py:195) ← MJCF <joint>. 두 링크를 잇는다")
    for j in sorted(robot.dynamic_joints, key=lambda x: x.idx):
        links = sorted(c.link_idx for c in j.connections)
        print(
            f"      dyna {j.idx:2d} ← joint '{dyna_to_joint.get(j.idx, '?')}': "
            f"{j.joint_type.value}, link {links[0]} ↔ link {links[1]}, "
            f"range={np.array(j.joint_range)}, armature={j.armature}"
        )

    print("\n    FixedJoint 토큰 (token.py:181) ← 관절 없이 붙은 body 사이")
    for j in sorted(robot.fixed_joints, key=lambda x: x.idx):
        links = sorted(c.link_idx for c in j.connections)
        print(f"      fixed {j.idx:2d}: link {links[0]} ↔ link {links[1]}")

    print("\n    Actuator 토큰 (token.py:232) ← MJCF <actuator>. dyna_joint 번호로 관절을 가리킨다")
    for a in sorted(robot.actuators, key=lambda x: x.idx):
        print(
            f"      act {a.idx} → dyna {a.dyna_joint_idx} ('{dyna_to_joint.get(a.dyna_joint_idx, '?')}'): "
            f"{a.actuator_type.value} kp={a.kp} kv={a.kv} force_range={np.array(a.force_range)}"
        )

    print("\n[4] serialize (t2/robotok/io.py:106) → 속성별 배열. zarr 에는 이 배열이 행으로 쌓인다")
    data = serialize(robot, include_states=False)
    for key in sorted(data):
        arr = np.asarray(data[key])
        print(f"      {key:28s} shape {arr.shape}")
    print("    같은 그룹의 속성을 알파벳 순으로 옆으로 붙이면 zarr 의 hardware/<그룹> 한 행이 된다")
    print("    (dyna_joint, fixed_joint 는 연결된 링크 2 개마다 1 행씩 → 관절 1 개 = 2 행)")


if __name__ == "__main__":
    main()
