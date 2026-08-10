"""
04d_prompt_smolvla.py - 같은 관측에 task 문자열만 바꿔보기 (VLA의 'L' 효과)

04b는 task 문자열을 거의 흘려보냈다 (데이터셋 값 또는 하드코딩 폴백).
그래서 SmolVLA의 'L'(언어)이 실제로 행동을 바꾸는지는 코드에서 안 보였다.

04d는 같은 프레임 한 장 (이미지 + 상태 동일)에서 task 문자열만 여러 개로
바꿔 가며 첫 액션이 얼마나 달라지는지 정량적으로 비교한다.
'V'와 'A' 사이에 끼어 있는 'L'이 뭘 하는지가 이 비교에서 가장 직관적으로 드러난다.

새로 도입되는 개념:
  - 언어 조건부 행동 (language-conditioned action)
  - baseline prompt 대비 L2 거리로 'L의 영향력' 정량화
  - 사전학습 모델의 prompt 민감도 (의미 있는 명령 vs 무관한 명령)

학습 포인트:
  - L2 거리 > 0  → task 문자열이 행동을 바꾸고 있다 = 'L' 작동
  - 의도된 prompt와 무관한 prompt 간 거리가 다르게 나오면, 모델이 단순히
    문자열을 무시하지 않고 의미를 일부 반영한다는 신호.
  - 사전학습 모델은 도메인 fine-tune 전이라 prompt 민감도가 약할 수 있다 →
    fine-tune이 왜 필요한지에 대한 또 다른 동기.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

# ============================================================
# 설정
# ============================================================
MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/svla_so100_pickplace"
FRAME_IDX = 0

# 비교할 task 문자열들 (첫 번째가 baseline)
PROMPTS = [
    "pick up the cube and place it in the box",   # baseline (의도된 명령)
    "pick up the cube and put it on the table",   # 의도 비슷, 목표 다름
    "move the arm to the left",                   # 다른 행동
    "stay still and do nothing",                  # 정적 명령
    "make a sandwich",                            # 완전 무관
    "",                                           # 빈 명령
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 1. 모델 + 전/후처리기 (04b와 동일)
# ============================================================
print(f"\nLoading model: {MODEL_ID}...")
policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
policy.eval()
policy.to(device)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
print("Model loaded!")

# ============================================================
# 2. 데이터셋 (한 프레임만)
# ============================================================
dataset = LeRobotDataset(DATASET_ID, episodes=[0])
frame = dataset[FRAME_IDX]
print(f"Using frame {FRAME_IDX} from {DATASET_ID}")

base_batch = {k: v.unsqueeze(0) for k, v in frame.items() if isinstance(v, torch.Tensor)}

# 이미지 키 매핑 (04b와 동일 로직)
model_img_keys = sorted([k for k in policy.config.input_features if "image" in k.lower()])
dataset_img_keys = sorted([k for k in base_batch if "image" in k.lower()])
for idx, m_key in enumerate(model_img_keys):
    if idx < len(dataset_img_keys):
        base_batch[m_key] = base_batch[dataset_img_keys[idx]]
    else:
        shape = policy.config.input_features[m_key].shape
        base_batch[m_key] = torch.zeros(1, *shape)
for k in dataset_img_keys:
    if k not in model_img_keys:
        del base_batch[k]

# ============================================================
# 3. prompt만 바꿔가며 첫 액션 받기
#    - reset()으로 내부 action chunk 큐를 비워야 매번 새 forward가 발생함
# ============================================================
results = []   # [(prompt, action_np), ...]

with torch.no_grad():
    for p in PROMPTS:
        policy.reset()
        batch = dict(base_batch)        # 같은 obs 재사용 (얕은 복사)
        batch["task"] = [p]
        processed = preprocessor(batch)
        action = policy.select_action(processed)
        action = postprocessor(action)
        results.append((p, action.squeeze().cpu().numpy()))

# 첫 prompt가 baseline
baseline_prompt, baseline_action = results[0]

# ============================================================
# 4. 출력 — prompt별 액션 + baseline과의 L2 거리
# ============================================================
print("\n== prompt별 첫 액션 비교 (앞 3 dim) | baseline 대비 L2 ==")
print("-" * 90)
print(f"{'prompt':50s} | {'action[:3]':>22s} | {'L2 vs base':>10s}")
print("-" * 90)
n = min(3, baseline_action.shape[-1])
for p, a in results:
    a_str = ", ".join(f"{v:6.3f}" for v in a[:n])
    l2 = float(np.linalg.norm(a - baseline_action))
    label = p if p else "(empty)"
    if len(label) > 50:
        label = label[:47] + "..."
    print(f"{label:50s} | [{a_str}] | {l2:10.4f}")
print("-" * 90)

# 차원별로 어디가 가장 많이 흔들렸는지 (모든 prompt에 걸쳐 std)
all_actions = np.stack([a for _, a in results], axis=0)   # (P, D)
per_dim_std = all_actions.std(axis=0)
print("\n== 차원별 prompt-induced 변동성 (std across prompts) ==")
for d, s in enumerate(per_dim_std):
    print(f"  dim {d}: {s:.4f}")

# ============================================================
# 5. paraphrase 비교 — '같은 의미 다른 표현' vs '다른 의미'
#    SmolVLA가 문자열의 표면형(syntax)을 보는지, 의미(semantics)를 보는지 검증.
#    같은 의미 prompt들 사이 평균 L2 (intra) < 다른 의미와의 평균 L2 (inter)
#    이면, 모델이 의미 단위로 행동을 결정하고 있다는 신호.
# ============================================================
PARAPHRASES = [
    "pick up the cube and place it in the box",     # baseline과 동일
    "put the cube into the box",
    "place the cube inside the box",
    "grab the cube and drop it in the box",
]
DISTRACTORS = [
    "move the arm to the left",
    "stay still and do nothing",
    "make a sandwich",
]

def get_action(task_str: str) -> np.ndarray:
    """같은 base_batch에 task만 바꿔 첫 액션을 받는 헬퍼."""
    policy.reset()
    batch = dict(base_batch)
    batch["task"] = [task_str]
    processed = preprocessor(batch)
    a = policy.select_action(processed)
    a = postprocessor(a)
    return a.squeeze().cpu().numpy()

with torch.no_grad():
    para_actions = [get_action(p) for p in PARAPHRASES]
    dist_actions = [get_action(d) for d in DISTRACTORS]

# intra: paraphrase ↔ paraphrase (모든 쌍)
intra = [
    float(np.linalg.norm(para_actions[i] - para_actions[j]))
    for i in range(len(PARAPHRASES))
    for j in range(i + 1, len(PARAPHRASES))
]
# inter: paraphrase × distractor (모든 쌍)
inter = [
    float(np.linalg.norm(pa - da))
    for pa in para_actions
    for da in dist_actions
]

intra_mean = float(np.mean(intra))
inter_mean = float(np.mean(inter))
ratio = inter_mean / max(intra_mean, 1e-9)

print("\n== Paraphrase vs Distractor 비교 ==")
print("  PARAPHRASES (같은 의미):")
for p in PARAPHRASES:
    print(f"    - {p!r}")
print("  DISTRACTORS (다른 의미):")
for d in DISTRACTORS:
    print(f"    - {d!r}")
print()
print(f"  intra  (paraphrase ↔ paraphrase) 평균 L2: {intra_mean:.4f}")
print(f"  inter  (paraphrase ↔ distractor)  평균 L2: {inter_mean:.4f}")
print(f"  ratio  (inter / intra)                  : {ratio:.2f}x")
print()
if inter_mean > intra_mean:
    print("  → inter > intra: 같은 의미끼리는 행동이 비슷하고, 다른 의미와는 더 멀다.")
    print("    모델이 표현(syntax)이 아니라 의미(semantics)에 반응하고 있다는 신호.")
else:
    print("  → inter ≤ intra: 의미 구분이 약하거나, prompt들이 사전학습 도메인에서")
    print("    멀어 신호 자체가 노이즈에 가까운 상태일 수 있음 → fine-tune의 동기.")

print("\nNote:")
print("  - L2 > 0  → task 문자열이 출력 액션에 실제로 영향을 준다는 직접 증거.")
print("  - 의도된 prompt 대비 무관한 prompt(예: 'make a sandwich')의 L2 거리가")
print("    얼마나 큰지 / 작은지 비교해 보면, 모델이 의미를 어느 정도 반영하는지")
print("    감을 잡을 수 있다.")
print("  - 빈 문자열은 '언어 신호 없음'에 가까운 베이스라인 — 다른 prompt와의")
print("    거리가 어느 정도인지가 'L 채널의 신호 강도' 추정치가 된다.")
print("\n추가 실험 아이디어:")
print("  · '빠르게 / 천천히' 같은 부사를 넣었을 때 액션 크기가 변하나?")
print("  · prompt를 영어 → 다른 언어로 바꿔보기 (모델 언어 backbone 확인)")
print("  · 04c의 chunk 비교와 결합: prompt별 50-step plan이 어떻게 갈라지나?")
print("  · PARAPHRASES/DISTRACTORS 풀을 키우고 t-test로 통계적 유의성 검정")
