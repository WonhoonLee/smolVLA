"""
01_test_install.py - SmolVLA 설치 확인 스크립트
로봇 없이 실행 가능
"""
import torch
import lerobot

print("=" * 50)
print("  SmolVLA / LeRobot 설치 확인")
print("=" * 50)

print(f"\nLeRobot version : {lerobot.__version__}")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM            : {mem / 1e9:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Device          : Apple Silicon (MPS)")
else:
    print("Device          : CPU only")

# SmolVLA 임포트 테스트
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("\nSmolVLAPolicy   : import OK")
except ImportError as e:
    print(f"\nSmolVLAPolicy   : import FAILED - {e}")

print("\nAll checks complete!")
