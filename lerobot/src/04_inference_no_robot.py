"""
04_inference_no_robot.py - 로봇 없이 데이터셋 프레임에서 모델 추론
사전학습 모델을 로드하여 기존 데이터셋의 프레임에 대해 행동을 예측하고,
실제 행동(ground truth)과 비교한다.
"""
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# ============================================================
# 설정
# ============================================================
MODEL_ID = "lerobot/diffusion_pusht"
DATASET_ID = "lerobot/pusht"
NUM_FRAMES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. 사전학습 모델 로드
# ============================================================
print(f"\nLoading model: {MODEL_ID}...")
policy = DiffusionPolicy.from_pretrained(MODEL_ID)
policy.eval()
policy.to(device)
print("Model loaded!")

# ============================================================
# 2. 데이터셋 로드
# ============================================================
print(f"Loading dataset: {DATASET_ID} (episode 0)...")
dataset = LeRobotDataset(DATASET_ID, episodes=[0])
print(f"Dataset: {dataset.num_frames} frames")

# ============================================================
# 3. 추론 실행 및 비교
# ============================================================
print(f"\nRunning inference on {NUM_FRAMES} frames...")
print("-" * 70)
print(f"{'Frame':>6s} | {'Predicted Action':>25s} | {'Ground Truth':>25s}")
print("-" * 70)

# Reset policy state
policy.reset()

with torch.no_grad():
    for i in range(min(NUM_FRAMES, len(dataset))):
        frame = dataset[i]

        # 배치 차원 추가 & GPU로 이동
        batch = {}
        for k, v in frame.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(device)

        # 행동 예측
        action = policy.select_action(batch)
        pred = action.squeeze().cpu().numpy()

        # 실제 행동 (ground truth)
        gt = frame["action"].numpy()

        # 출력
        pred_str = ", ".join(f"{v:7.1f}" for v in pred[:2])
        gt_str = ", ".join(f"{v:7.1f}" for v in gt[:2])
        print(f"{i:6d} | [{pred_str}] | [{gt_str}]")

print("-" * 70)
print("\nInference complete!")
print("Note: 예측값과 GT가 다른 것은 정상입니다.")
print("      Diffusion Policy는 확률적으로 행동을 생성합니다.")
