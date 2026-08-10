"""
02_explore_dataset.py - 공개 데이터셋 탐색 (로봇 없이 실행 가능)
HuggingFace Hub에서 LeRobot 데이터셋을 다운로드하여 살펴본다.
"""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================
# 1. 데이터셋 로드 (에피소드 0~2만)
# ============================================================
print("=" * 50)
print("  PushT 데이터셋 탐색")
print("=" * 50)

print("\nLoading dataset (episodes 0, 1, 2)...")
dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

print(f"총 프레임 수   : {dataset.num_frames}")
print(f"FPS            : {dataset.fps}")

# ============================================================
# 2. 단일 프레임 살펴보기
# ============================================================
print("\n" + "=" * 50)
print("  프레임 0 상세 정보")
print("=" * 50)

frame = dataset[0]
print(f"키 목록: {list(frame.keys())}")

for key, value in frame.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key:30s} | shape={str(value.shape):20s} | dtype={value.dtype}")
    else:
        print(f"  {key:30s} | value={value}")

# ============================================================
# 3. 여러 프레임의 action 분포 확인
# ============================================================
print("\n" + "=" * 50)
print("  Action 통계 (전체 프레임)")
print("=" * 50)

actions = torch.stack([dataset[i]["action"] for i in range(dataset.num_frames)])
print(f"  Action shape : {actions.shape}")
print(f"  Action mean  : {actions.mean(dim=0).numpy()}")
print(f"  Action std   : {actions.std(dim=0).numpy()}")
print(f"  Action min   : {actions.min(dim=0).values.numpy()}")
print(f"  Action max   : {actions.max(dim=0).values.numpy()}")

# ============================================================
# 4. 이미지 시각화 (matplotlib)
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if "observation.image" in frame:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        fig.suptitle("PushT Dataset - Episode 0, Frames 0~4", fontsize=14)

        for i in range(5):
            img = dataset[i]["observation.image"]
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            img = img.numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis("off")

        plt.tight_layout()
        out_path = "pusht_frames_sample.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n이미지 저장 완료: {out_path}")
    else:
        print("\n이 데이터셋에는 이미지 관측이 없습니다.")
except ImportError:
    print("\nmatplotlib가 설치되지 않아 시각화를 건너뜁니다.")

print("\n탐색 완료!")
