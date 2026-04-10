"""
06_finetune_smolvla.py - SmolVLA Fine-tuning (Python API)
수집한 데이터셋 또는 공개 데이터셋으로 SmolVLA를 fine-tuning한다.
GPU 필요 (16GB+ VRAM 권장).
"""
import torch
from pathlib import Path
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

# ============================================================
# 설정
# ============================================================
BASE_MODEL = "lerobot/smolvla_base"
DATASET_ID = "your_user/koch_pick_place"  # 수집한 데이터셋 ID
OUTPUT_DIR = Path("outputs/train/smolvla_finetuned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4          # 24GB VRAM 기준, 부족하면 2로 줄이기
NUM_STEPS = 100000
LR = 1e-4

device = torch.device("cuda")
print(f"Using device: {device}")

# ============================================================
# 1. 모델 로드
# ============================================================
print(f"Loading base model: {BASE_MODEL}...")
policy = SmolVLAPolicy.from_pretrained(BASE_MODEL)
policy.to(device)
policy.train()

preprocess, postprocess = make_pre_post_processors(
    policy.config, BASE_MODEL,
    preprocessor_overrides={
        "device_processor": {"device": str(device)}
    },
)
print("Model loaded!")

# ============================================================
# 2. 데이터셋 로드
# ============================================================
print(f"Loading dataset: {DATASET_ID}...")
dataset = LeRobotDataset(DATASET_ID)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
print(f"Dataset: {dataset.num_frames} frames")

# ============================================================
# 3. 옵티마이저 설정
# ============================================================
optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

# ============================================================
# 4. 학습 루프
# ============================================================
print(f"\nTraining for {NUM_STEPS} steps...")
print("-" * 50)

step = 0
best_loss = float("inf")

while step < NUM_STEPS:
    for batch in dataloader:
        batch = preprocess(batch)
        loss, outputs = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step:7d}/{NUM_STEPS} | Loss: {loss.item():.4f}")

        # 주기적 체크포인트 저장
        if step % 10000 == 0 and step > 0:
            ckpt_dir = OUTPUT_DIR / f"checkpoint_{step}"
            policy.save_pretrained(ckpt_dir)
            print(f"  >> Checkpoint saved: {ckpt_dir}")

        if loss.item() < best_loss:
            best_loss = loss.item()

        step += 1
        if step >= NUM_STEPS:
            break

# ============================================================
# 5. 최종 모델 저장
# ============================================================
print(f"\nSaving final model to {OUTPUT_DIR}...")
policy.save_pretrained(OUTPUT_DIR)
print(f"Training complete! Best loss: {best_loss:.4f}")
