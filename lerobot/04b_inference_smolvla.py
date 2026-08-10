"""
04b_inference_smolvla.py - SmolVLA로 데이터셋 프레임에서 추론
04_inference_no_robot.py의 SmolVLA 버전.

로봇/시뮬레이터 없이, 공개 LeRobot 데이터셋의 프레임을 SmolVLA에 넣어
액션을 예측하고 실제 액션(ground truth)과 비교한다.

04(DiffusionPolicy)와의 핵심 차이 (학습 포인트):
  1. 모델:        DiffusionPolicy -> SmolVLAPolicy
  2. 전/후처리:    수동 정규화 X -> make_pre_post_processors가 처리
  3. 언어 명령:    없음 -> "task" 문자열 필수 (VLA의 'L')
  4. 액션:        2D (pushT) -> 6D (so100 robot)
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ============================================================
# 설정
# ============================================================
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so100_pickplace"
NUM_FRAMES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. 사전학습 모델 + 전/후처리기 로드
# ============================================================
print(f"\nLoading model: {MODEL_ID}...")
policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
policy.eval()
policy.to(device)

# 전처리기: (정규화, 토큰화, 디바이스 이동)을 한 번에 처리
# 후처리기: (역정규화, CPU 이동)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=MODEL_ID,
    preprocessor_overrides={
        "device_processor": {"device": str(device)}
    },
)
print("Model loaded!")

# 모델이 어떤 입력 키를 기대하는지 확인 (디버깅용)
print("\nModel input features:")
for k, v in policy.config.input_features.items():
    print(f"  {k}: shape={tuple(v.shape)}")
print(f"Action steps per call: {policy.config.n_action_steps} (chunk)")

# ============================================================
# 2. 데이터셋 로드
# ============================================================
print(f"\nLoading dataset: {DATASET_ID} (episode 0)...")
dataset = LeRobotDataset(DATASET_ID, episodes=[0])
print(f"Dataset: {dataset.num_frames} frames")

# ============================================================
# 3. 추론 + Ground Truth 비교
# ============================================================
print(f"\nRunning inference on {NUM_FRAMES} frames...")
print("-" * 80)
print(f"{'Frame':>5s} | {'Predicted (first 3 dims)':>30s} | {'Ground Truth':>30s}")
print("-" * 80)

# 모델이 기대하는 이미지 키 (camera1/2/3) ≠ 데이터셋 키 (top/wrist)
# → 매핑 필요
model_img_keys = sorted([k for k in policy.config.input_features if "image" in k.lower()])

policy.reset()

with torch.no_grad():
    for i in range(min(NUM_FRAMES, len(dataset))):
        frame = dataset[i]

        # 1) 텐서 필드는 배치 차원 추가
        batch = {}
        for k, v in frame.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)  #batch 차원 추가

        # 2) 이미지 키 매핑: 데이터셋 키 → 모델이 기대하는 키
        #    데이터셋의 카메라 수가 모델보다 적으면 부족분은 0으로 채움
        dataset_img_keys = sorted([k for k in batch if "image" in k.lower()])
        for idx, m_key in enumerate(model_img_keys):
            if idx < len(dataset_img_keys):
                #데이터셋 이름(image_top)을 모델 이름(camera1)으로 alias(별칭) 붙이는 작업
                batch[m_key] = batch[dataset_img_keys[idx]] 
            else:
                shape = policy.config.input_features[m_key].shape
                batch[m_key] = torch.zeros(1, *shape)
        # 모델이 모르는 데이터셋 키는 제거
        for k in dataset_img_keys:
            if k not in model_img_keys:
                del batch[k]

        # 3) task 문자열 — SmolVLA는 자연어 명령이 필수
        if "task" in frame and isinstance(frame["task"], str):
            batch["task"] = [frame["task"]]
        else:
            batch["task"] = ["pick up the cube and place it in the box\n"]

        # 4) 전처리 → 추론 → 후처리
        processed = preprocessor(batch)  #사람 데이터 → AI가 이해하는 언어
        action = policy.select_action(processed) #추론 단계
        action = postprocessor(action) #AI 출력 → 사람이 이해하는 언어

        pred = action.squeeze().cpu().numpy() #모델이 예측한 action tensor를 NumPy 배열로 표현
        gt = frame["action"].numpy()

        # 출력 (앞 3개 차원만)
        n = min(3, len(pred), len(gt))
        pred_str = ", ".join(f"{v:7.3f}" for v in pred[:n]) # [ -0.415,   0.972,   0.080]
        gt_str = ", ".join(f"{v:7.3f}" for v in gt[:n])
        print(f"{i:5d} | [{pred_str}] | [{gt_str}]")

print("-" * 80)
print("\nInference complete!")
print("Note:")
print("  - SmolVLA는 한 번의 forward로 50-step action chunk를 예측합니다.")
print("    select_action()은 chunk에서 다음 한 스텝씩 꺼내서 반환합니다.")
print("  - 예측과 GT가 항상 일치하지는 않습니다. 사전학습 모델을 그대로 쓴 것이고,")
print("    fine-tune을 하면 특정 데이터셋에 더 잘 맞춰집니다.")
