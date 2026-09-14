"""모델 학습 한 step 을 실제 데이터로 단계별 출력한다.

    MUJOCO_GL=egl python explore/step2_train_trace.py

데이터: 4 단계 평가 결과 zarr (학습 데이터와 같은 형식) 를 explore/out/trace_data.zarr 로 복사해 쓴다.
모델:   로봇 설계용 체크포인트 wheeled_bimanual/mgoc83ra (hardware_gen) 를 불러와 학습을 몇 step 더 해 본다.
원본 체크포인트 파일은 바꾸지 않는다 (메모리 안에서만 가중치를 고친다).
"""

import argparse
import pickle
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import default_collate

import t2.utils.misc  # noqa: F401  (${eval:...} resolver 등록)
from t2.model.modality import DiffusionModalityAdapter
from t2.model.t2 import setup_decoder

CKPT = "checkpoints/wheeled_bimanual/mgoc83ra/035.pt"
TRACE_ZARR = Path("explore/out/trace_data.zarr")


def title(text):
    print(f"\n{'=' * 90}\n{text}\n{'=' * 90}")


def prepare_zarr(src: str):
    """평가 결과 zarr 를 복사하고 target_pose 를 학습 데이터 형식 (pos 3 + rotmat 9) 으로 맞춘다."""
    if TRACE_ZARR.exists():
        shutil.rmtree(TRACE_ZARR)
    idx = Path(str(TRACE_ZARR).replace(".zarr", ".zarr.idx"))
    if idx.exists():
        shutil.rmtree(idx)
    shutil.copytree(src, TRACE_ZARR)
    root = zarr.open_group(str(TRACE_ZARR), mode="r+")
    arr = root["rollout/target_pose"]
    if dict(arr.attrs) != {"pos": 3, "rotmat": 9}:
        data = arr[:][..., :12]
        del root["rollout/target_pose"]
        new = root.create_array("rollout/target_pose", data=data)
        new.attrs.update({"pos": 3, "rotmat": 9})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="wandb/latest-run/files/ctrl_eval_summary.zarr")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=5)
    args = parser.parse_args()
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=150)
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    device = torch.device("cuda")

    # ------------------------------------------------------------------ [1]
    title("[1] 준비: 체크포인트 설정(cfg.pkl) 과 모델 불러오기")
    cfg = pickle.load(open(Path(CKPT).parent / "cfg.pkl", "rb"))
    OmegaConf.set_struct(cfg, False)
    seq_len = OmegaConf.to_container(cfg.seq_len, resolve=True)
    print("학습한 task:", list(cfg.model.tasks.keys()))
    print("design space 최대 행 수 (seq_len):", seq_len)
    bundle = setup_decoder(
        cfg, ckpt_path=CKPT, device=device, task_name="hardware_gen", disable_grad=False,
        use_flash_attn=False, use_mixed_precision=False, use_torch_compile=False,
    )
    model, task = bundle.model, bundle.task_model
    model.train()
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ------------------------------------------------------------------ [2]
    title("[2] zarr 목차 → 학습 샘플 번호표 만들기 (T2Dataset.__init__)")
    prepare_zarr(args.zarr)
    ds_cfg = cfg.datasets.clean
    dataset = hydra.utils.instantiate(ds_cfg.dataset, path=str(TRACE_ZARR), use_pbar=False)
    root = zarr.open(str(TRACE_ZARR), mode="r")
    print("zarr:", TRACE_ZARR)
    print("  로봇 수:", root["hardware_meta/seed"].shape[0], " episode 수:", root["rollout_meta/ends"].shape[0],
          " 전체 step:", root["rollout/ctrl"].shape[0])
    print("학습 샘플 수 (= 길이 100 이상인 episode 의 step 수):", len(dataset))
    print("샘플 번호표 한 줄 (13 칸) = [로봇 seed, actuator 시작/끝 행, dyna_joint 시작/끝, fixed_joint 시작/끝,"
          " link 시작/끝, episode seed, 현재 step, episode 시작/끝]")
    print("  예: 샘플 0 =", np.asarray(dataset.indices[0]).tolist())

    # ------------------------------------------------------------------ [3]
    title("[3] 샘플 하나 꺼내기 (T2Dataset.__getitem__)")
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(len(dataset), size=args.batch_size, replace=False)
    sample = dataset[int(sample_ids[0])]
    row = np.asarray(dataset.indices[int(sample_ids[0])]).astype(int)
    print(f"샘플 {sample_ids[0]}: 로봇 link 행 {row[7]}~{row[8] - 1}, episode 행 {row[11]}~{row[12] - 1}, 현재 step {row[10]}")
    print("  몸 토큰 (hardware 에서 그 로봇 행을 잘라 옴):")
    for k in ["link/mass", "link/geom_size", "dyna_joint/link/id", "actuator/kp"]:
        print(f"    {k:28s} shape {sample[k].shape}  앞 3 행 {sample[k][:3].reshape(3, -1).tolist()}")
    print("  움직임 토큰 (그 episode 에서 무작위 8 개 시점을 잘라 옴):")
    for k in ["target_pose/pos", "track_link_obs/pos", "dyna_joint_obs/qpos", "actuator_obs/force", "ctrl/target_qpos"]:
        print(f"    {k:28s} shape {sample[k].shape}  (시점 8 × 개수, 칸)")
    time_ids = sample["dyna_joint_obs/time/id"].reshape(seq_len["rollout_steps"], -1)[:, 0]
    print("    time/id (기준 시점에서 몇 step 차이):", time_ids.astype(int).tolist())

    # ------------------------------------------------------------------ [4]
    title(f"[4] 샘플 {args.batch_size} 개를 배치로 묶고 번호표 붙이기 (batch_process_fn)")
    batch = default_collate([dataset[int(i)] for i in sample_ids])
    batch = {k: v.to(device) for k, v in batch.items()}
    before = set(batch)
    process = hydra.utils.instantiate(ds_cfg.batch_process_fn)
    batch = process(batch)
    print("추가된 번호표:")
    for k in sorted(set(batch) - before):
        v = batch[k][0].flatten()
        print(f"  {k:30s} shape {tuple(batch[k].shape)}  앞 12 개 {v[:12].tolist()}")

    # ------------------------------------------------------------------ [5]
    title("[5] 토큰 종류별 역할: 조건(보여 줌) vs 맞힐 것(노이즈) — hardware_gen task")
    print(f"{'토큰 종류':18s} {'역할':10s} {'토큰 수':>8s} {'한 토큰 칸 수(인코딩 후)':>24s}   번호표")
    total = 0
    for name, adapter in sorted(task.adapters.items()):
        role = "맞힐 것" if adapter.loss_weight > 0 else "조건"
        n = adapter.modality.max_seq_len
        total += n
        print(f"{name:18s} {role:10s} {n:8d} {adapter.modality.dim:24d}   {list(adapter.modality.id_attrs)}")
    print(f"샘플 하나의 토큰 수 합계: {total}  (+ 확산 단계 t 정보)")

    # ------------------------------------------------------------------ [6]
    title("[6] 노이즈 섞기 예시 — link 토큰 (DiffusionModalityAdapter.prepare_tokens)")
    link = task.adapters["link"]
    assert isinstance(link, DiffusionModalityAdapter)
    clean = link.modality.to_tensor(batch, encoders=link.attr_encoders, normalize=True)
    print("깨끗한 link 토큰 (정규화·인코딩 후) shape:", tuple(clean.shape), " 첫 토큰 앞 8 칸:", clean[0, 0, :8])
    sched = task.noise_scheduler
    for t in [5, 50, 95]:
        noise = torch.randn_like(clean)
        noisy = sched.add_noise(clean, noise, torch.full((clean.shape[0],), t, device=device))
        a = sched.alphas_cumprod[t].sqrt().item()
        print(f"  t={t:2d}: 노이즈 섞인 값 = {a:.2f} × 깨끗한 값 + {np.sqrt(1 - a * a):.2f} × 노이즈"
              f"  → 첫 토큰 앞 8 칸 {noisy[0, 0, :8]}")
    print("모델이 맞혀야 하는 정답 = 섞은 노이즈 자체")

    # ------------------------------------------------------------------ [7]
    title(f"[7] 학습 step: 노이즈 맞히기 → loss → 가중치 수정 ({args.train_steps} 번 반복, 같은 배치·같은 노이즈)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    for step in range(args.train_steps):
        torch.manual_seed(123)  # 매 step 같은 t, 같은 노이즈 → loss 변화만 보이도록
        losses = model(batches=[(["hardware_gen"], dict(batch))], is_eval=False)
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        parts = {k.split("/")[-1]: round(v.item(), 4) for k, v in losses.items() if k != "loss"}
        print(f"  step {step}: 전체 loss {losses['loss'].item():.4f}   토큰별 {parts}")
    print("\n실제 학습은 이 step 을 서로 다른 배치(1024 개씩), 무작위 t·노이즈로 수만 번 반복한다.")


if __name__ == "__main__":
    main()
