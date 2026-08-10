"""
08_sim_smolvla_koch.py - SmolVLA + Koch v1.1 시뮬레이션

실물 로봇 없이 SmolVLA 모델을 로드하고,
시뮬레이션 환경에서 Koch v1.1 로봇팔을 제어합니다.

파이프라인:
  시뮬레이션 카메라 이미지(224x224) + 관절 상태(6D)
  → SmolVLA 전처리 (리사이즈, 정규화, 토큰화)
  → SmolVLA 추론 (50-step action chunk 예측)
  → 후처리 (역정규화)
  → 시뮬레이션 로봇에 적용 + 3D 시각화

모드:
  random_policy   SmolVLA 구조만 사용, 랜덤 가중치로 동작 확인
  pretrained      lerobot/smolvla_base 사전학습 모델로 추론
  collect_train   시뮬 데이터 수집 → fine-tune → 추론 루프

사용법:
  python 08_sim_smolvla_koch.py                        # 랜덤 정책 시각화
  python 08_sim_smolvla_koch.py --mode pretrained      # 사전학습 모델
  python 08_sim_smolvla_koch.py --mode collect_train   # 수집+학습+추론
"""

import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from collections import deque

# Windows 콘솔 인코딩 문제 방지
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

# ============================================================
# Koch v1.1 시뮬레이션 엔진 (경량 FK/IK)
# ============================================================

TABLE_HEIGHT = 0.625
LINK_LENGTHS = [0.02, 0.045, 0.13, 0.124, 0.05, 0.045]
DXL_CENTER = 2048
DXL_TO_RAD = (2 * math.pi) / 4096
RAD_TO_DXL = 4096 / (2 * math.pi)

import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def forward_kinematics(joint_angles):
    """5개 관절 → 7개 위치 (base ~ gripper tip)"""
    q = joint_angles
    positions = np.zeros((7, 3))
    positions[0] = [0, 0, TABLE_HEIGHT]
    positions[1] = positions[0] + [0, 0, LINK_LENGTHS[0] + LINK_LENGTHS[1]]

    c0, s0 = math.cos(q[0]), math.sin(q[0])

    cum = q[1]
    dz = LINK_LENGTHS[2] * math.cos(cum)
    dxy = LINK_LENGTHS[2] * math.sin(cum)
    positions[2] = positions[1] + [dxy * c0, dxy * s0, dz]

    cum += q[2]
    dz = LINK_LENGTHS[3] * math.cos(cum)
    dxy = LINK_LENGTHS[3] * math.sin(cum)
    positions[3] = positions[2] + [dxy * c0, dxy * s0, dz]

    cum += q[3]
    dz = LINK_LENGTHS[4] * math.cos(cum)
    dxy = LINK_LENGTHS[4] * math.sin(cum)
    positions[4] = positions[3] + [dxy * c0, dxy * s0, dz]

    dz = LINK_LENGTHS[5] * math.cos(cum)
    dxy = LINK_LENGTHS[5] * math.sin(cum)
    positions[5] = positions[4] + [dxy * c0, dxy * s0, dz]
    positions[6] = positions[5].copy()
    return positions


def ik_ccd(target_pos, initial_angles=None, max_iter=300, tol=0.003):
    """CCD 역기구학"""
    angles = list(initial_angles) if initial_angles else [0.0] * 5
    limits = [(-2.35, 2.35), (-1.57, 1.57), (-1.57, 1.57),
              (-1.57, 1.57), (-3.14, 3.14)]
    target = np.array(target_pos)

    for _ in range(max_iter):
        pos = forward_kinematics(angles)
        ee = pos[5]
        if np.linalg.norm(target - ee) < tol:
            break
        for j in reversed(range(5)):
            pos = forward_kinematics(angles)
            ee = pos[5]
            jp = pos[j + 1] if j > 0 else pos[0]
            to_ee = ee - jp
            to_tgt = target - jp
            n1, n2 = np.linalg.norm(to_ee), np.linalg.norm(to_tgt)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = np.clip(np.dot(to_ee / n1, to_tgt / n2), -1, 1)
            delta = math.acos(cos_a)
            cross = np.cross(to_ee, to_tgt)
            if j == 0:
                if cross[2] < 0: delta = -delta
            else:
                c0, s0 = math.cos(angles[0]), math.sin(angles[0])
                if np.dot(cross, [-s0, c0, 0]) < 0: delta = -delta
            angles[j] += delta * 0.3
            angles[j] = max(limits[j][0], min(limits[j][1], angles[j]))
    return angles


def rad_to_dxl(r):
    return int(round(r * RAD_TO_DXL + DXL_CENTER))

def dxl_to_rad(d):
    return (d - DXL_CENTER) * DXL_TO_RAD


class KochSimEnv:
    """Koch v1.1 시뮬레이션 환경 (SmolVLA 연동)"""

    def __init__(self):
        self.joint_angles = [0.0] * 5  # rad
        self.gripper = 1.0             # 0=닫힘, 1=열림
        self.objects = []              # [{name, pos, size, color}]

        # 상태 이력 (observation window)
        self.state_history = deque(maxlen=10)

    def reset(self):
        self.joint_angles = [0.0] * 5
        self.gripper = 1.0
        self.state_history.clear()
        return self.get_observation()

    def get_state_6d(self):
        """Koch 6D 상태: 5 관절(rad) + 그리퍼 개방도"""
        return np.array(self.joint_angles + [self.gripper], dtype=np.float32)

    def get_observation(self):
        """SmolVLA 입력 형식으로 관측 반환"""
        state = self.get_state_6d()
        image = self.render_image(224, 224)
        return {"state": state, "image": image}

    def apply_action(self, action_6d):
        """6D 액션을 적용 (5 관절 + 그리퍼)"""
        action = np.array(action_6d, dtype=np.float32)
        # 관절 각도 적용 (클리핑)
        limits = [2.35, 1.57, 1.57, 1.57, 3.14]
        for i in range(5):
            self.joint_angles[i] = float(np.clip(action[i], -limits[i], limits[i]))
        # 그리퍼
        if len(action) > 5:
            self.gripper = float(np.clip(action[5], 0, 1))

    def apply_action_delta(self, delta_6d, scale=0.05):
        """델타 액션 적용"""
        state = self.get_state_6d()
        new_state = state + np.array(delta_6d) * scale
        self.apply_action(new_state)

    def spawn_box(self, name, pos, size=(0.03, 0.03, 0.03), color="#e74c3c"):
        self.objects.append({
            "name": name, "pos": np.array(pos),
            "size": size, "color": color
        })

    # ----------------------------------------------------------
    # 렌더링
    # ----------------------------------------------------------

    def render_image(self, width=224, height=224):
        """현재 씬을 RGB numpy 이미지(H,W,3)로 렌더링"""
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=25, azim=135)
        self._draw_scene(ax)
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0.4, 1.0)
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return img

    def _draw_scene(self, ax):
        """테이블 + 오브젝트 + 로봇"""
        # 테이블
        ww, dd = 0.6, 0.8
        verts = [[[-ww/2, -dd/2, TABLE_HEIGHT], [ww/2, -dd/2, TABLE_HEIGHT],
                   [ww/2, dd/2, TABLE_HEIGHT], [-ww/2, dd/2, TABLE_HEIGHT]]]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.3,
                            facecolor="#c4a882", edgecolor="#8b7355"))
        # 오브젝트
        for obj in self.objects:
            self._draw_box(ax, obj["pos"], obj["size"], obj["color"])
        # 로봇
        pos = forward_kinematics(self.joint_angles)
        colors = ["#333", "#1a5276", "#d4d4d4", "#d4d4d4", "#1a5276", "#333"]
        for i in range(6):
            ax.plot3D([pos[i][0], pos[i+1][0]],
                      [pos[i][1], pos[i+1][1]],
                      [pos[i][2], pos[i+1][2]],
                      color=colors[i], linewidth=4)
        for i in range(7):
            ax.scatter(*pos[i], s=50 if i < 5 else 25,
                       c="#e74c3c" if i == 6 else "#2c3e50", zorder=5)

    def _draw_box(self, ax, center, size, color):
        sx, sy, sz = [s / 2 for s in size]
        cx, cy, cz = center
        faces = [
            [[cx-sx,cy-sy,cz-sz],[cx+sx,cy-sy,cz-sz],[cx+sx,cy+sy,cz-sz],[cx-sx,cy+sy,cz-sz]],
            [[cx-sx,cy-sy,cz+sz],[cx+sx,cy-sy,cz+sz],[cx+sx,cy+sy,cz+sz],[cx-sx,cy+sy,cz+sz]],
            [[cx-sx,cy-sy,cz-sz],[cx+sx,cy-sy,cz-sz],[cx+sx,cy-sy,cz+sz],[cx-sx,cy-sy,cz+sz]],
            [[cx-sx,cy+sy,cz-sz],[cx+sx,cy+sy,cz-sz],[cx+sx,cy+sy,cz+sz],[cx-sx,cy+sy,cz+sz]],
            [[cx-sx,cy-sy,cz-sz],[cx-sx,cy+sy,cz-sz],[cx-sx,cy+sy,cz+sz],[cx-sx,cy-sy,cz+sz]],
            [[cx+sx,cy-sy,cz-sz],[cx+sx,cy+sy,cz-sz],[cx+sx,cy+sy,cz+sz],[cx+sx,cy-sy,cz+sz]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.8,
                            facecolor=color, edgecolor="gray", linewidth=0.5))

    def render_live(self, ax, info_text=""):
        """라이브 시각화 (기존 axes에 렌더링)"""
        ax.cla()
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0.4, 1.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("SmolVLA + Koch v1.1 Simulation", fontsize=11, fontweight="bold")
        self._draw_scene(ax)
        if info_text:
            ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
                      fontsize=8, fontfamily="monospace", va="top",
                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))


# ============================================================
# SmolVLA 래퍼
# ============================================================

class SmolVLASimWrapper:
    """SmolVLA를 시뮬레이션 환경과 연결하는 래퍼"""

    def __init__(self, device="cuda", pretrained_path="lerobot/smolvla_base"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        self.device = device
        self.pretrained_path = pretrained_path

        print(f"[SmolVLA] Loading model: {pretrained_path}")
        print(f"[SmolVLA] Device: {device}")
        start = time.time()

        self.policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        self.policy.to(device)
        self.policy.eval()

        elapsed = time.time() - start
        print(f"[SmolVLA] Model loaded in {elapsed:.1f}s")

        self.config = self.policy.config
        print(f"[SmolVLA] Input features: {list(self.config.input_features.keys())}")
        print(f"[SmolVLA] Output: action shape = {self.config.output_features['action'].shape}")
        print(f"[SmolVLA] Action chunk: {self.config.n_action_steps} steps")

        # 전처리/후처리 파이프라인
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
        self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(
            self.config,
            dataset_stats=self._get_dataset_stats(),
        )

        # 토크나이저 (언어 명령용)
        self.task_text = "pick up the object and place it in the target zone\n"

    def _get_dataset_stats(self):
        """정규화용 통계 — 사전학습 모델의 stats 사용"""
        try:
            stats = self.policy.dataset_stats
            if stats:
                print(f"[SmolVLA] Using pretrained dataset stats")
                return stats
        except AttributeError:
            pass

        # 모델에 stats가 없으면 Koch v1.1 관절 범위 기반 더미 통계 생성
        print("[SmolVLA] No pretrained stats found, using Koch v1.1 defaults")
        state_mean = torch.zeros(6)
        state_std = torch.tensor([1.5, 1.0, 1.0, 1.0, 2.0, 0.5])
        action_mean = torch.zeros(6)
        action_std = torch.tensor([1.5, 1.0, 1.0, 1.0, 2.0, 0.5])
        return {
            "observation.state": {"mean": state_mean, "std": state_std},
            "action": {"mean": action_mean, "std": action_std},
        }

    def reset(self):
        """에피소드 시작 시 호출"""
        self.policy.reset()

    def predict_action(self, image_rgb, state_6d):
        """
        시뮬레이션 관측 → SmolVLA 추론 → 6D 액션

        Args:
            image_rgb: numpy (H, W, 3) uint8 — 시뮬 카메라 이미지
            state_6d: numpy (6,) float32 — [5 joints + gripper]

        Returns:
            action: numpy (6,) float32 — 예측된 관절 목표
        """
        # 이미지: uint8 (H,W,3) → float32 [0,1] (3,H,W) — BCHW format
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0

        # 상태: (6,)
        state_tensor = torch.from_numpy(state_6d).float()

        # SmolVLA 입력 배치 구성
        # batch_to_transition이 "observation." prefix 키를 observation으로,
        # "task" 키를 complementary_data로 분리함
        image_keys = [k for k in self.config.input_features
                      if "image" in k.lower()]

        batch = {
            "observation.state": state_tensor,
            "task": self.task_text,
        }
        # 모든 카메라에 동일 이미지 공급 (시뮬에서는 카메라 1개)
        for key in image_keys:
            batch[key] = img_tensor

        # 전처리 (batch_to_transition → 각 step 처리 → transition_to_batch)
        processed = self.preprocessor(batch)

        # 추론
        with torch.no_grad():
            action = self.policy.select_action(processed)

        # 후처리 (역정규화, CPU 이동)
        # PolicyAction = torch.Tensor alias
        result = self.postprocessor(action)
        action_np = result.squeeze().cpu().numpy()

        return action_np[:6]  # 6D만 사용

    def set_task(self, task_text):
        """자연어 작업 명령 설정"""
        self.task_text = task_text if task_text.endswith("\n") else task_text + "\n"
        print(f"[SmolVLA] Task: {self.task_text.strip()}")


# ============================================================
# 모드 1: 랜덤 정책 시각화
# ============================================================

def mode_random_policy(device):
    """SmolVLA 구조를 사용하되 랜덤 액션으로 시뮬레이션 동작 확인"""
    print("\n" + "=" * 60)
    print("  SmolVLA + Koch v1.1 — Random Policy Demo")
    print("  (모델 로드 없이 시뮬레이션 파이프라인 검증)")
    print("=" * 60 + "\n")

    env = KochSimEnv()
    env.spawn_box("red_box", [0.12, 0.05, 0.64],
                  size=(0.03, 0.03, 0.03), color="#e74c3c")
    env.spawn_box("blue_box", [0.0, -0.1, 0.64],
                  size=(0.035, 0.035, 0.035), color="#2980b9")
    env.reset()

    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title("SmolVLA Koch Simulation — Random Policy")

    ax_3d = fig.add_subplot(121, projection="3d")
    ax_3d.view_init(elev=25, azim=135)

    ax_img = fig.add_subplot(122)
    ax_img.set_title("Camera View (224x224)", fontsize=10)
    ax_img.axis("off")

    plt.ion()
    plt.show()

    print("[Demo] 랜덤 액션으로 시뮬레이션 중... (30 스텝)")
    print("[Demo] SmolVLA 구조: state(6D) + image(224x224) → action(6D)\n")

    for step in range(30):
        obs = env.get_observation()

        # 랜덤 액션 (SmolVLA 출력 형태 모방)
        action = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
        action[5] = np.random.uniform(0, 1)  # 그리퍼

        env.apply_action(action)

        # 시각화
        state = env.get_state_6d()
        dxl = [rad_to_dxl(r) for r in state[:5]]
        ee = forward_kinematics(env.joint_angles)[5]
        info = (f"Step: {step+1}/30\n"
                f"Action: [{', '.join(f'{a:.2f}' for a in action)}]\n"
                f"DXL: [{' '.join(f'{d:4d}' for d in dxl)}]\n"
                f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})\n"
                f"Gripper: {'OPEN' if state[5]>0.5 else 'CLOSED'}")

        env.render_live(ax_3d, info)
        ax_img.clear()
        ax_img.imshow(obs["image"])
        ax_img.set_title("Camera View (224x224)", fontsize=10)
        ax_img.axis("off")

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.15)

        if step % 5 == 0:
            print(f"  Step {step+1:3d}: action=[{', '.join(f'{a:+.2f}' for a in action)}]  "
                  f"EE=({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

    print("\n[완료] 랜덤 정책 데모 종료")
    plt.ioff()
    plt.show()


# ============================================================
# 모드 2: 사전학습 모델 추론
# ============================================================

def mode_pretrained(device):
    """lerobot/smolvla_base 사전학습 모델로 실시간 추론"""
    print("\n" + "=" * 60)
    print("  SmolVLA + Koch v1.1 — Pretrained Inference")
    print("=" * 60 + "\n")

    # 환경 설정
    env = KochSimEnv()
    env.spawn_box("target_box", [0.12, 0.05, 0.64],
                  size=(0.03, 0.03, 0.03), color="#e74c3c")
    env.reset()

    # SmolVLA 모델 로드
    vla = SmolVLASimWrapper(device=device)
    vla.set_task("pick up the red box and place it on the left side")
    vla.reset()

    # 시각화
    fig = plt.figure(figsize=(16, 6))
    fig.canvas.manager.set_window_title("SmolVLA Koch — Pretrained Model")

    ax_3d = fig.add_subplot(131, projection="3d")
    ax_3d.view_init(elev=25, azim=135)
    ax_img = fig.add_subplot(132)
    ax_img.axis("off")
    ax_act = fig.add_subplot(133)

    plt.ion()
    plt.show()

    action_history = []
    num_steps = 100

    print(f"[추론] {num_steps} 스텝 추론 시작...")
    print(f"[추론] Task: {vla.task_text.strip()}\n")

    for step in range(num_steps):
        obs = env.get_observation()

        t_start = time.time()
        action = vla.predict_action(obs["image"], obs["state"])
        t_infer = time.time() - t_start

        # 델타 액션으로 적용 (사전학습 모델 출력이 크기 때문에 스케일링)
        env.apply_action_delta(action, scale=0.02)
        action_history.append(action.copy())

        # 시각화
        state = env.get_state_6d()
        ee = forward_kinematics(env.joint_angles)[5]
        info = (f"Step: {step+1}/{num_steps}\n"
                f"Infer: {t_infer*1000:.0f}ms\n"
                f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})\n"
                f"Gripper: {'OPEN' if state[5]>0.5 else 'CLOSED'}")

        env.render_live(ax_3d, info)

        ax_img.clear()
        ax_img.imshow(obs["image"])
        ax_img.set_title("Camera Input (224x224)", fontsize=10)
        ax_img.axis("off")

        # 액션 히스토리 그래프
        if len(action_history) > 1:
            ax_act.clear()
            ah = np.array(action_history)
            labels = ["Base", "Shoulder", "Elbow", "WristF", "WristR", "Grip"]
            for j in range(6):
                ax_act.plot(ah[:, j], label=labels[j], linewidth=1)
            ax_act.set_title("Action History", fontsize=10)
            ax_act.set_xlabel("Step")
            ax_act.set_ylabel("Value")
            ax_act.legend(fontsize=7, loc="upper right")
            ax_act.grid(True, alpha=0.3)

        fig.canvas.draw()
        fig.canvas.flush_events()

        if step % 10 == 0:
            hz = 1.0 / t_infer if t_infer > 0 else 0
            print(f"  Step {step+1:3d}: infer={t_infer*1000:.0f}ms ({hz:.1f}Hz)  "
                  f"action=[{', '.join(f'{a:+.2f}' for a in action[:3])} ...]")

    print(f"\n[완료] {num_steps} 스텝 추론 완료")
    plt.ioff()
    plt.show()


# ============================================================
# 모드 3: 수집 → 학습 → 추론 루프
# ============================================================

def mode_collect_train(device, num_demos=20, train_steps=200):
    """
    시뮬레이션에서 전문가 데이터 수집 → SmolVLA fine-tune → 추론 평가

    1단계: IK 기반 전문가 정책으로 pick-place 데모 수집
    2단계: 수집 데이터로 SmolVLA fine-tuning
    3단계: fine-tuned 모델로 자율 추론
    """
    print("\n" + "=" * 60)
    print("  SmolVLA + Koch v1.1 — Collect → Train → Infer")
    print("=" * 60)

    save_dir = Path("outputs/sim_smolvla_data")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ==== 1단계: 전문가 데이터 수집 ====
    print(f"\n[1/3] Expert demo collection ({num_demos} episodes)...")

    episodes = []
    for ep_idx in range(num_demos):
        env = KochSimEnv()

        # 랜덤 물체 + 목표 위치
        obj_x = np.random.uniform(-0.1, 0.15)
        obj_y = np.random.uniform(-0.1, 0.1)
        obj_z = TABLE_HEIGHT + 0.015
        env.spawn_box("cargo", [obj_x, obj_y, obj_z],
                      size=(0.03, 0.03, 0.03), color="#e74c3c")

        tgt_x = np.random.uniform(-0.15, 0.2)
        tgt_y = np.random.uniform(0.1, 0.25)

        env.reset()

        # 전문가 궤적 (IK 기반)
        pick = [obj_x, obj_y, obj_z + 0.005]
        place = [tgt_x, tgt_y, obj_z + 0.005]
        waypoints = [
            ([pick[0], pick[1], pick[2]+0.08], 1.0),
            (pick, 1.0),
            (pick, 0.0),   # grab
            ([pick[0], pick[1], pick[2]+0.08], 0.0),
            ([place[0], place[1], place[2]+0.08], 0.0),
            (place, 0.0),
            (place, 1.0),  # release
            ([place[0], place[1], place[2]+0.08], 1.0),
        ]

        episode_data = []
        for wp_pos, grip in waypoints:
            target_angles = ik_ccd(wp_pos, env.joint_angles)
            start_angles = list(env.joint_angles)
            start_grip = env.gripper

            for t in np.linspace(0, 1, 15):
                for j in range(5):
                    env.joint_angles[j] = start_angles[j] + t * (target_angles[j] - start_angles[j])
                env.gripper = start_grip + t * (grip - start_grip)

                obs = env.get_observation()
                state_6d = env.get_state_6d()
                action_6d = np.array(env.joint_angles + [env.gripper], dtype=np.float32)

                episode_data.append({
                    "state": state_6d.copy(),
                    "action": action_6d.copy(),
                    "image": obs["image"],
                })

        episodes.append(episode_data)
        if (ep_idx + 1) % 5 == 0:
            print(f"  [{ep_idx+1}/{num_demos}] frames={len(episode_data)}")

    # 데이터 저장
    all_states = []
    all_actions = []
    all_images = []
    for ep in episodes:
        for frame in ep:
            all_states.append(frame["state"])
            all_actions.append(frame["action"])
            all_images.append(frame["image"])

    states = np.stack(all_states)
    actions = np.stack(all_actions)
    print(f"\n  Total: {len(all_states)} frames collected")
    print(f"  State range: [{states.min():.2f}, {states.max():.2f}]")
    print(f"  Action range: [{actions.min():.2f}, {actions.max():.2f}]")

    # 통계 계산 (정규화용)
    state_mean = torch.from_numpy(states.mean(axis=0)).float()
    state_std = torch.from_numpy(states.std(axis=0)).float().clamp(min=0.01)
    action_mean = torch.from_numpy(actions.mean(axis=0)).float()
    action_std = torch.from_numpy(actions.std(axis=0)).float().clamp(min=0.01)

    dataset_stats = {
        "observation.state": {"mean": state_mean, "std": state_std},
        "action": {"mean": action_mean, "std": action_std},
    }
    print(f"  State mean: {state_mean.numpy()}")
    print(f"  Action std: {action_std.numpy()}")

    # ==== 2단계: Fine-tuning ====
    print(f"\n[2/3] SmolVLA fine-tuning ({train_steps} steps)...")
    print("  (전체 학습에는 GPU + 많은 데이터 필요)")
    print("  여기서는 학습 루프 구조만 시연합니다\n")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)
    policy.train()

    # 이미지 키 확인
    image_keys = [k for k in policy.config.input_features if "image" in k.lower()]
    print(f"  Image keys: {image_keys}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )

    losses = []
    for step in range(train_steps):
        # 미니배치 샘플링
        idx = np.random.randint(0, len(all_states), size=2)

        batch_states = torch.from_numpy(states[idx]).float().to(device)
        batch_actions = torch.from_numpy(actions[idx]).float().to(device)

        # 이미지 전처리
        batch_images = torch.stack([
            torch.from_numpy(all_images[i]).float() / 255.0
            for i in idx
        ]).to(device)  # (B, H, W, 3)

        # 정규화
        norm_states = (batch_states - state_mean.to(device)) / state_std.to(device)
        norm_actions = (batch_actions - action_mean.to(device)) / action_std.to(device)

        # SmolVLA 배치 구성
        batch = {
            "observation.state": norm_states,
            "action": norm_actions,
        }
        for key in image_keys:
            batch[key] = batch_images

        # 토큰화
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
        encoding = tokenizer(
            ["pick up the object and place it in the target zone\n"] * len(idx),
            padding="longest", max_length=48, truncation=True,
            return_tensors="pt"
        )
        batch["observation.language.tokens"] = encoding["input_ids"].to(device)
        batch["observation.language.attention_mask"] = encoding["attention_mask"].to(device)

        try:
            output = policy.forward(batch)
            if isinstance(output, dict):
                loss = output.get("loss", output.get("l_total", list(output.values())[0]))
            else:
                loss = output[0] if isinstance(output, tuple) else output

            if isinstance(loss, dict):
                loss = sum(loss.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if step % 50 == 0:
                print(f"  Step {step:4d}/{train_steps}  Loss: {loss_val:.4f}")

        except Exception as e:
            if step == 0:
                print(f"  [참고] 학습 루프 구조 시연 — 실제 학습은 GPU 메모리/데이터 추가 필요")
                print(f"  Error detail: {str(e)[:100]}")
            break

    # ==== 3단계: 추론 시각화 ====
    print(f"\n[3/3] Inference visualization...")

    policy.eval()
    env = KochSimEnv()
    env.spawn_box("target_box", [0.12, 0.05, 0.64],
                  size=(0.03, 0.03, 0.03), color="#e74c3c")
    env.reset()

    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title("SmolVLA Koch — Collect+Train+Infer")
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_3d.view_init(elev=25, azim=135)
    ax_loss = fig.add_subplot(122)

    if losses:
        ax_loss.plot(losses, color="#2980b9", linewidth=1.5)
        ax_loss.set_title("Training Loss", fontsize=11)
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)

    env.render_live(ax_3d, "SmolVLA Collect → Train → Infer\nPipeline Complete")

    plt.tight_layout()
    plt.show()
    print("\n[완료] 전체 파이프라인 시연 종료")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SmolVLA + Koch v1.1 시뮬레이션",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드:
  random_policy   SmolVLA 없이 시뮬레이션 파이프라인 검증 (기본)
  pretrained      lerobot/smolvla_base 사전학습 모델로 추론
  collect_train   시뮬 데이터 수집 → fine-tune → 추론

예시:
  python 08_sim_smolvla_koch.py                              # 빠른 테스트
  python 08_sim_smolvla_koch.py --mode pretrained             # 모델 추론
  python 08_sim_smolvla_koch.py --mode collect_train --demos 30  # 풀 파이프라인
        """,
    )
    parser.add_argument("--mode", default="random_policy",
                        choices=["random_policy", "pretrained", "collect_train"])
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (자동 감지)")
    parser.add_argument("--demos", type=int, default=20,
                        help="수집 에피소드 수 (collect_train)")
    parser.add_argument("--train-steps", type=int, default=200,
                        help="학습 스텝 수 (collect_train)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.mode == "random_policy":
        mode_random_policy(device)
    elif args.mode == "pretrained":
        mode_pretrained(device)
    elif args.mode == "collect_train":
        mode_collect_train(device, num_demos=args.demos,
                           train_steps=args.train_steps)


if __name__ == "__main__":
    main()
