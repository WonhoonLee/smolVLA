"""
03_train_sim_diffusion.py - PushT 시뮬레이션에서 Diffusion Policy 학습
로봇 없이 실행 가능. GPU 권장.
pip install "lerobot[pusht]" 필요.
"""
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ============================================================
# 설정
# ============================================================
DATASET_ID = "lerobot/pusht"
OUTPUT_DIR = Path("outputs/train/diffusion_pusht")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8          # 8GB VRAM용
NUM_STEPS = 300         # 데모용 (본 학습 시 100000 권장)
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. 사전학습 모델 로드 (config 확인용)
# ============================================================
print("Loading pretrained Diffusion Policy...")
policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht")
cfg = policy.config
print(f"  horizon={cfg.horizon}, n_obs_steps={cfg.n_obs_steps}, n_action_steps={cfg.n_action_steps}")

# ============================================================
# 2. 데이터셋 로드 (모델의 temporal window에 맞게)
# ============================================================
print("Loading dataset with delta_timestamps...")
fps = 10  # PushT fps
dt = 1.0 / fps

# n_obs_steps=2 -> 관측 2프레임, horizon=16 -> 행동 16프레임
delta_timestamps = {
    "observation.image": [dt * (i - cfg.n_obs_steps + 1) for i in range(cfg.n_obs_steps)],
    "observation.state": [dt * (i - cfg.n_obs_steps + 1) for i in range(cfg.n_obs_steps)],
    "action": [dt * (i - cfg.n_obs_steps + 1) for i in range(cfg.horizon)],
}

dataset = LeRobotDataset(DATASET_ID, delta_timestamps=delta_timestamps)
print(f"Total frames: {dataset.num_frames}")

# 데이터 검증
sample = dataset[0]
print(f"  observation.image shape: {sample['observation.image'].shape}")
print(f"  observation.state shape: {sample['observation.state'].shape}")
print(f"  action shape: {sample['action'].shape}")

# ============================================================
# 3. 데이터 로더
# ============================================================
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,       # Windows에서는 0
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)

# ============================================================
# 4. 학습 설정
# ============================================================
policy.train()
policy.to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

print(f"\nTraining for {NUM_STEPS} steps (batch_size={BATCH_SIZE})...")
print("-" * 50)

# ============================================================
# 5. 학습 루프
# ============================================================
step = 0
while step < NUM_STEPS:
    for batch in dataloader:
        # GPU로 이동
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device)
            else:
                batch_gpu[k] = v

        # action_is_pad 추가
        if "action_is_pad" not in batch_gpu and "action" in batch_gpu:
            batch_gpu["action_is_pad"] = torch.zeros(
                batch_gpu["action"].shape[:-1],
                dtype=torch.bool, device=device
            )

        loss, _ = policy.forward(batch_gpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  Step {step:5d} / {NUM_STEPS} | Loss: {loss.item():.4f}")

        step += 1
        if step >= NUM_STEPS:
            break

# ============================================================
# 6. 모델 저장
# ============================================================
print(f"\nSaving model to {OUTPUT_DIR}...")
policy.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Checkpoint at: {OUTPUT_DIR}")
