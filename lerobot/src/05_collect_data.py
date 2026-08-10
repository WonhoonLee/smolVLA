"""
05_collect_data.py - Koch v1.1 텔레오퍼레이션으로 시연 데이터 수집
*** 실제 Koch v1.1 로봇팔 필요 ***
"""
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.koch_follower import KochFollower, KochFollowerConfig
from lerobot.teleoperators.koch_leader import KochLeader, KochLeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.processor import make_default_processors

# ============================================================
# 설정 (자신의 환경에 맞게 수정)
# ============================================================
FOLLOWER_PORT = "/dev/ttyACM1"      # lerobot-find-port로 확인
LEADER_PORT = "/dev/ttyACM0"
CAMERA_INDEX = 0                     # 웹캠 인덱스
HF_REPO_ID = "your_user/koch_pick_place"  # HuggingFace 사용자명 수정
TASK_DESC = "Pick up the block and place it in the bin"
NUM_EPISODES = 50
FPS = 30
EPISODE_LENGTH_SEC = 60

# ============================================================
# 로봇 설정
# ============================================================
robot_config = KochFollowerConfig(
    port=FOLLOWER_PORT,
    id="my_koch_follower",
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=640, height=480, fps=FPS,
        )
    },
)
teleop_config = KochLeaderConfig(
    port=LEADER_PORT,
    id="my_koch_leader",
)

robot = KochFollower(robot_config)
teleop = KochLeader(teleop_config)

# ============================================================
# 데이터셋 생성
# ============================================================
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")

dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features={**action_features, **obs_features},
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# ============================================================
# 녹화 시작
# ============================================================
_, events = init_keyboard_listener()
robot.connect()
teleop.connect()

teleop_proc, robot_act_proc, robot_obs_proc = make_default_processors()

print(f"\n{'=' * 50}")
print(f"  데이터 수집 시작: {NUM_EPISODES} 에피소드")
print(f"  작업: {TASK_DESC}")
print(f"{'=' * 50}")
print("  Enter: 다음 에피소드 시작")
print("  Backspace: 현재 에피소드 삭제 후 재녹화")
print("  Ctrl+C: 종료\n")

for ep in range(NUM_EPISODES):
    print(f"\n--- Episode {ep + 1}/{NUM_EPISODES} ---")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_proc,
        robot_action_processor=robot_act_proc,
        robot_observation_processor=robot_obs_proc,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_LENGTH_SEC,
        single_task=TASK_DESC,
    )
    dataset.save_episode()

robot.disconnect()
teleop.disconnect()

# HuggingFace Hub에 업로드
print("\nUploading dataset to HuggingFace Hub...")
dataset.push_to_hub()
print(f"Dataset uploaded: https://huggingface.co/datasets/{HF_REPO_ID}")
