"""
07_run_policy_koch.py - 학습된 SmolVLA로 Koch v1.1 자율 제어
*** 실제 Koch v1.1 로봇팔 필요 ***
"""
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener

# ============================================================
# 설정 (자신의 환경에 맞게 수정)
# ============================================================
MODEL_ID = "your_user/smolvla_koch_pick_place"
FOLLOWER_PORT = "/dev/ttyACM1"
CAMERA_INDEX = 0
TASK_DESC = "Pick up the block and place it in the bin"
EVAL_REPO_ID = "your_user/eval_smolvla_koch"
NUM_EVAL_EPISODES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. 로봇 초기화
# ============================================================
robot_config = KochFollowerConfig(
    port=FOLLOWER_PORT,
    id="my_koch_follower",
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=640, height=480, fps=30,
        )
    },
)
robot = KochFollower(robot_config)

# ============================================================
# 2. 모델 로드
# ============================================================
print(f"Loading model: {MODEL_ID}...")
policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
policy = policy.to(device).eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=MODEL_ID,
    preprocessor_overrides={
        "device_processor": {"device": str(device)}
    },
)
print("Model loaded!")

# ============================================================
# 3. 평가 데이터셋 (로깅용)
# ============================================================
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")

dataset = LeRobotDataset.create(
    repo_id=EVAL_REPO_ID,
    fps=30,
    features={**action_features, **obs_features},
    robot_type=robot.name,
    use_videos=True,
)

# ============================================================
# 4. 자율 제어 실행
# ============================================================
_, events = init_keyboard_listener()
robot.connect()

print(f"\n{'=' * 50}")
print(f"  자율 제어 시작: {NUM_EVAL_EPISODES} 에피소드")
print(f"  작업: {TASK_DESC}")
print(f"{'=' * 50}")
print("  Ctrl+C: 비상 정지\n")

successes = 0
for ep in range(NUM_EVAL_EPISODES):
    print(f"\n--- Eval Episode {ep + 1}/{NUM_EVAL_EPISODES} ---")

    record_loop(
        robot=robot,
        events=events,
        fps=30,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=60,
        single_task=TASK_DESC,
        display_data=True,
    )
    dataset.save_episode()

    # 수동 성공 여부 기록
    result = input("  성공 여부 (y/n): ").strip().lower()
    if result == "y":
        successes += 1

robot.disconnect()

# 결과 출력
print(f"\n{'=' * 50}")
print(f"  결과: {successes}/{NUM_EVAL_EPISODES} 성공")
print(f"  성공률: {successes / NUM_EVAL_EPISODES * 100:.1f}%")
print(f"{'=' * 50}")

# Hub 업로드
dataset.push_to_hub()
print(f"Evaluation data uploaded: {EVAL_REPO_ID}")
