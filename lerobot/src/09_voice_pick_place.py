"""
09_voice_pick_place.py - 음성/텍스트 명령으로 Koch v1.1 Pick & Place 시뮬레이션

음성(또는 키보드) 명령으로 물체를 잡아서 박스에 넣는 시뮬레이션.
실물 로봇 없이 PC만으로 실행 가능.

파이프라인:
  음성/텍스트 명령 → 자연어 파싱 (색상/물체/목표 추출)
  → IK 기반 Pick & Place 실행 → 3D 시각화

모드:
  voice    마이크 음성 인식 (speech_recognition + Google STT)
  text     키보드 텍스트 입력 (기본값, 패키지 불필요)
  auto     사전 정의된 명령 자동 실행 (데모용)

사용법:
  python 09_voice_pick_place.py                    # 텍스트 입력 모드
  python 09_voice_pick_place.py --mode voice       # 음성 인식 모드
  python 09_voice_pick_place.py --mode auto        # 자동 데모
  python 09_voice_pick_place.py --lang ko          # 한국어 음성 인식

필요 패키지:
  pip install numpy matplotlib                     # 기본 (text/auto 모드)
  pip install SpeechRecognition pyaudio            # voice 모드용 추가
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button

# Windows 호환
import platform
if platform.system() == "Windows":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ================================================================
# Koch v1.1 시뮬레이션 엔진 (FK/IK + Pick & Place 물리)
# ================================================================

TABLE_HEIGHT = 0.625
LINK_LENGTHS = [0.02, 0.045, 0.13, 0.124, 0.05, 0.045]


@dataclass
class SimObject:
    """시뮬레이션 오브젝트"""
    name: str
    position: np.ndarray
    size: tuple = (0.03, 0.03, 0.03)
    color: str = "#e74c3c"
    label: str = ""
    picked: bool = False
    is_container: bool = False  # 박스/통 여부


def forward_kinematics(joint_angles):
    """5개 관절 -> 7개 위치"""
    q = joint_angles
    pos = np.zeros((7, 3))
    pos[0] = [0, 0, TABLE_HEIGHT]
    pos[1] = pos[0] + [0, 0, LINK_LENGTHS[0] + LINK_LENGTHS[1]]

    c0, s0 = math.cos(q[0]), math.sin(q[0])

    cum = q[1]
    dz = LINK_LENGTHS[2] * math.cos(cum)
    dxy = LINK_LENGTHS[2] * math.sin(cum)
    pos[2] = pos[1] + [dxy * c0, dxy * s0, dz]

    cum += q[2]
    dz = LINK_LENGTHS[3] * math.cos(cum)
    dxy = LINK_LENGTHS[3] * math.sin(cum)
    pos[3] = pos[2] + [dxy * c0, dxy * s0, dz]

    cum += q[3]
    dz = LINK_LENGTHS[4] * math.cos(cum)
    dxy = LINK_LENGTHS[4] * math.sin(cum)
    pos[4] = pos[3] + [dxy * c0, dxy * s0, dz]

    dz = LINK_LENGTHS[5] * math.cos(cum)
    dxy = LINK_LENGTHS[5] * math.sin(cum)
    pos[5] = pos[4] + [dxy * c0, dxy * s0, dz]
    pos[6] = pos[5].copy()
    return pos


def _ik_ccd_single(target, init_angles, max_iter=600, tol=0.002):
    """단일 초기값 CCD"""
    angles = list(init_angles)
    limits = [(-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57),
              (-1.57, 1.57), (-3.14, 3.14)]

    for iteration in range(max_iter):
        p = forward_kinematics(angles)
        err = np.linalg.norm(target - p[5])
        if err < tol:
            break
        damping = 0.5 if iteration < 200 else 0.3
        for j in reversed(range(5)):
            p = forward_kinematics(angles)
            ee, jp = p[5], p[j + 1] if j > 0 else p[0]
            to_ee, to_tgt = ee - jp, target - jp
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
            angles[j] += delta * damping
            angles[j] = max(limits[j][0], min(limits[j][1], angles[j]))

    err = np.linalg.norm(target - forward_kinematics(angles)[5])
    return angles, err


def ik_ccd(target_pos, initial_angles=None, max_iter=600, tol=0.002):
    """CCD 역기구학 (multi-start: 여러 초기값 중 최적 선택)"""
    target = np.array(target_pos)
    base_angle = math.atan2(target[1], target[0]) if (abs(target[0]) > 1e-6 or abs(target[1]) > 1e-6) else 0.0

    # 여러 초기 자세 후보
    candidates = [
        [base_angle, 0.5, -0.5, -0.3, 0.0],   # 앞으로 뻗은 자세
        [base_angle, 0.8, -0.8, -0.2, 0.0],    # 더 앞으로
        [base_angle, 0.3, -0.3, -0.5, 0.0],    # 아래로
        [base_angle, 1.0, -1.0, 0.0, 0.0],     # 크게 뻗기
        [base_angle, 0.0, 0.0, 0.0, 0.0],      # 직립
    ]
    if initial_angles:
        init = list(initial_angles)
        init[0] = base_angle
        candidates.insert(0, init)  # 현재 자세도 후보에 포함

    best_angles = None
    best_err = float("inf")

    for init in candidates:
        angles, err = _ik_ccd_single(target, init, max_iter, tol)
        if err < best_err:
            best_err = err
            best_angles = angles
        if err < tol:
            break

    return best_angles


class PickPlaceSimulator:
    """Koch v1.1 Pick & Place 시뮬레이션 (음성 명령 연동)"""

    def __init__(self):
        self.joint_angles = [0.0] * 5
        self.gripper = 1.0  # 1=open, 0=closed
        self.objects: list[SimObject] = []
        self.picked_object = None
        self.command_log = []

    # --- 오브젝트 관리 ---

    def add_object(self, name, position, size=(0.03, 0.03, 0.03),
                   color="#e74c3c", label="", is_container=False):
        obj = SimObject(
            name=name, position=np.array(position, dtype=float),
            size=size, color=color, label=label or name,
            is_container=is_container,
        )
        self.objects.append(obj)
        return obj

    def find_object(self, keyword):
        """키워드로 오브젝트 검색 (이름/라벨/색상)"""
        keyword = keyword.lower().strip()
        # 색상 매핑 (한국어 + 영어)
        color_map = {
            "빨간": "red", "빨강": "red", "red": "red",
            "파란": "blue", "파랑": "blue", "blue": "blue",
            "초록": "green", "녹색": "green", "green": "green",
            "노란": "yellow", "노랑": "yellow", "yellow": "yellow",
            "보라": "purple", "purple": "purple",
        }
        # 컨테이너 키워드
        container_words = ["박스", "box", "상자", "통", "bin", "바구니", "basket", "컨테이너"]

        # 1. 정확한 이름 매칭
        for obj in self.objects:
            if keyword in obj.name.lower() or keyword in obj.label.lower():
                return obj

        # 2. 색상 매칭
        for color_key, color_val in color_map.items():
            if color_key in keyword:
                for obj in self.objects:
                    if color_val in obj.name.lower() or color_val in obj.color.lower():
                        return obj

        # 3. 컨테이너 매칭
        for cw in container_words:
            if cw in keyword:
                for obj in self.objects:
                    if obj.is_container:
                        return obj

        return None

    # --- 물리 로직 ---

    def try_pick(self):
        """그리퍼 근처 물체 잡기"""
        if self.gripper > 0.3:
            return False
        ee = forward_kinematics(self.joint_angles)[5]
        for obj in self.objects:
            if not obj.picked and not obj.is_container:
                dist = np.linalg.norm(ee[:2] - obj.position[:2])  # XY 평면 거리
                if dist < 0.08:
                    obj.picked = True
                    self.picked_object = obj
                    return True
        return False

    def try_place(self):
        """잡은 물체 놓기"""
        if self.picked_object and self.gripper > 0.5:
            self.picked_object.position = forward_kinematics(self.joint_angles)[5].copy()
            self.picked_object.picked = False
            self.picked_object = None
            return True
        return False

    def update_picked(self):
        """잡은 물체를 EE 위치로 업데이트"""
        if self.picked_object:
            self.picked_object.position = forward_kinematics(self.joint_angles)[5].copy()

    # --- 동작 실행 ---

    def go_home(self):
        self.joint_angles = [0.0] * 5
        self.gripper = 1.0

    def execute_pick_and_place(self, pick_obj, place_obj, draw_callback=None):
        """Pick & Place 시퀀스 실행"""
        pick_pos = pick_obj.position.copy()
        pick_pos[2] = TABLE_HEIGHT + 0.08  # 테이블 위 8cm

        if place_obj.is_container:
            place_pos = place_obj.position.copy()
            place_pos[2] = TABLE_HEIGHT + 0.10
        else:
            place_pos = place_obj.position.copy()
            place_pos[2] = TABLE_HEIGHT + 0.08

        hover = 0.08
        pick_above = [pick_pos[0], pick_pos[1], pick_pos[2] + hover]
        place_above = [place_pos[0], place_pos[1], place_pos[2] + hover]

        steps = 50
        delay = 0.02

        # 물체 방향으로 base 회전 초기값 설정 (IK 수렴 개선)
        base_angle = math.atan2(pick_pos[1], pick_pos[0])
        self.joint_angles[0] = base_angle

        # 중간 높은 위치 (긴 거리 이동 시 IK 안정성 확보)
        mid_height = TABLE_HEIGHT + 0.18
        mid_point = [
            (pick_pos[0] + place_pos[0]) / 2,
            (pick_pos[1] + place_pos[1]) / 2,
            mid_height,
        ]

        phases = [
            (pick_above, True, "접근 중...", None),
            (pick_pos, True, "하강 중...", None),
            (None, False, "잡는 중...", None),          # pick
            (pick_above, False, "상승 중...", None),
            (mid_point, False, "이동 중...", place_pos), # 중간점 경유
            (place_above, False, "목표 접근...", None),
            (place_pos, False, "하강 중...", None),
            (None, True, "놓는 중...", None),            # place
            (place_above, True, "복귀 중...", None),
        ]

        for target, grip_open, phase_label, base_hint in phases:
            if draw_callback:
                draw_callback(phase_label)

            # base 회전 힌트 (place 이동 전 방향 전환)
            if base_hint is not None:
                self.joint_angles[0] = math.atan2(base_hint[1], base_hint[0])

            if target is None:
                # pick 또는 place 동작
                if not grip_open:
                    self.gripper = 0.0
                    self.try_pick()
                else:
                    self.gripper = 1.0
                    self.try_place()
                for _ in range(12):
                    self.update_picked()
                    if draw_callback:
                        draw_callback(phase_label)
                    time.sleep(delay)
                continue

            target_angles = ik_ccd(target, self.joint_angles)
            start_angles = list(self.joint_angles)
            start_grip = self.gripper
            target_grip = 1.0 if grip_open else 0.0

            for i in range(steps):
                t = (i + 1) / steps
                for j in range(5):
                    self.joint_angles[j] = start_angles[j] + t * (target_angles[j] - start_angles[j])
                self.gripper = start_grip + t * (target_grip - start_grip)
                self.update_picked()
                if draw_callback:
                    draw_callback(phase_label)
                time.sleep(delay)

    # --- 3D 렌더링 ---

    def render(self, ax, status_text="", command_text=""):
        ax.cla()
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.35, 0.35)
        ax.set_zlim(0.4, 1.0)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Koch v1.1 Voice Pick & Place", fontsize=12, fontweight="bold")

        # 테이블
        w, d = 0.6, 0.8
        h = TABLE_HEIGHT
        verts = [[[-w/2, -d/2, h], [w/2, -d/2, h],
                   [w/2, d/2, h], [-w/2, d/2, h]]]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.3,
                            facecolor="#c4a882", edgecolor="#8b7355"))

        # 오브젝트
        for obj in self.objects:
            if not obj.picked:
                alpha = 0.4 if obj.is_container else 0.85
                self._draw_box(ax, obj.position, obj.size, obj.color, alpha)
                ax.text(obj.position[0], obj.position[1],
                        obj.position[2] + obj.size[2] / 2 + 0.015,
                        obj.label, fontsize=7, ha="center", color=obj.color,
                        fontweight="bold")

        # 잡고 있는 물체
        if self.picked_object:
            self._draw_box(ax, self.picked_object.position,
                           self.picked_object.size,
                           self.picked_object.color, 0.6)

        # 로봇
        pos = forward_kinematics(self.joint_angles)
        colors = ["#333", "#1a5276", "#d4d4d4", "#d4d4d4", "#1a5276", "#333"]
        for i in range(6):
            ax.plot3D([pos[i][0], pos[i+1][0]],
                      [pos[i][1], pos[i+1][1]],
                      [pos[i][2], pos[i+1][2]],
                      color=colors[i], linewidth=4, solid_capstyle="round")
        for i in range(7):
            s = 50 if i < 5 else 25
            c = "#e74c3c" if i == 6 else "#2c3e50"
            ax.scatter(*pos[i], s=s, c=c, zorder=5, depthshade=True)

        # 상태 텍스트
        ee = pos[5]
        info = f"EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})\n"
        info += f"Gripper: {'OPEN' if self.gripper > 0.5 else 'CLOSED'}\n"
        if status_text:
            info += f"\n{status_text}"
        ax.text2D(0.02, 0.95, info, transform=ax.transAxes,
                  fontsize=8, va="top",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 명령 텍스트
        if command_text:
            ax.text2D(0.5, 0.02, command_text, transform=ax.transAxes,
                      fontsize=11, ha="center", fontweight="bold",
                      color="#2c3e50",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="#e8f8f5", edgecolor="#1abc9c"))

    def _draw_box(self, ax, center, size, color, alpha=0.8):
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
        ax.add_collection3d(Poly3DCollection(faces, alpha=alpha,
                            facecolor=color, edgecolor="gray", linewidth=0.5))


# ================================================================
# 자연어 명령 파서
# ================================================================

class CommandParser:
    """음성/텍스트 명령을 파싱하여 (pick_target, place_target) 추출"""

    # 색상 키워드 (한/영)
    COLOR_KW = {
        "빨간": "red", "빨강": "red", "빨간색": "red", "red": "red",
        "파란": "blue", "파랑": "blue", "파란색": "blue", "blue": "blue",
        "초록": "green", "녹색": "green", "초록색": "green", "green": "green",
        "노란": "yellow", "노랑": "yellow", "노란색": "yellow", "yellow": "yellow",
        "보라": "purple", "보라색": "purple", "purple": "purple",
    }

    # 물체 키워드
    OBJECT_KW = ["물체", "물건", "블록", "block", "cube", "object", "것", "거"]

    # 컨테이너 키워드
    CONTAINER_KW = ["박스", "box", "상자", "통", "bin", "바구니", "basket",
                     "컨테이너", "container"]

    # 동작 키워드
    PICK_KW = ["잡아", "집어", "들어", "pick", "grab", "잡고", "집고"]
    PLACE_KW = ["넣어", "놓아", "놓고", "place", "put", "drop", "옮겨", "이동"]

    def parse(self, text, sim):
        """
        명령 텍스트 → (pick_object, place_object) 또는 None

        예시:
          "빨간 블록을 박스에 넣어" → (red_block, box)
          "pick up the blue object and put it in the box"
          "파란 물체를 상자에 옮겨"
        """
        text = text.lower().strip()
        if not text:
            return None

        pick_target = None
        place_target = None

        # 1. 색상으로 pick 대상 찾기
        for kw, color in self.COLOR_KW.items():
            if kw in text:
                pick_target = sim.find_object(color)
                if pick_target and not pick_target.is_container:
                    break
                pick_target = None

        # 2. 색상 없으면 물체 키워드로 찾기
        if pick_target is None:
            for kw in self.OBJECT_KW:
                if kw in text:
                    for obj in sim.objects:
                        if not obj.is_container:
                            pick_target = obj
                            break
                    break

        # 3. 컨테이너(place 대상) 찾기
        for kw in self.CONTAINER_KW:
            if kw in text:
                place_target = sim.find_object(kw)
                break

        # 4. place 대상이 없으면 기본 컨테이너
        if place_target is None:
            for obj in sim.objects:
                if obj.is_container:
                    place_target = obj
                    break

        # 5. pick 대상이 없으면 첫 번째 비컨테이너 물체
        if pick_target is None:
            for obj in sim.objects:
                if not obj.is_container and not obj.picked:
                    pick_target = obj
                    break

        if pick_target and place_target:
            return (pick_target, place_target)
        return None


# ================================================================
# 음성 인식
# ================================================================

class VoiceInput:
    """마이크 음성 인식 (speech_recognition 사용)"""

    def __init__(self, lang="ko-KR"):
        self.lang = lang
        self.recognizer = None
        self.microphone = None
        self.available = False

        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.available = True
            # 노이즈 조정
            with self.microphone as source:
                print("[음성] 주변 소음 보정 중... (1초)")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print(f"[음성] 마이크 준비 완료 (언어: {lang})")
        except ImportError:
            print("[음성] speech_recognition 미설치. 텍스트 입력으로 전환합니다.")
            print("       설치: pip install SpeechRecognition pyaudio")
        except OSError as e:
            print(f"[음성] 마이크 접근 실패: {e}")
            print("       텍스트 입력으로 전환합니다.")

    def listen(self, timeout=5):
        """음성 인식 → 텍스트 반환 (실패 시 None)"""
        if not self.available:
            return None

        import speech_recognition as sr

        print("\n[음성] 말씀하세요... ", end="", flush=True)
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout,
                                                phrase_time_limit=8)
            text = self.recognizer.recognize_google(audio, language=self.lang)
            print(f'"{text}"')
            return text
        except sr.WaitTimeoutError:
            print("(시간 초과)")
            return None
        except sr.UnknownValueError:
            print("(인식 실패)")
            return None
        except sr.RequestError as e:
            print(f"(API 오류: {e})")
            return None


# ================================================================
# 메인 실행 모드
# ================================================================

def setup_scene(sim):
    """시뮬레이션 씬 구성: 물체 3개 + 박스 1개"""
    # 물체들 (테이블 위)
    sim.add_object("red_block", [0.12, 0.0, TABLE_HEIGHT + 0.015],
                   size=(0.03, 0.03, 0.03), color="#e74c3c", label="RED")
    sim.add_object("blue_block", [-0.08, 0.05, TABLE_HEIGHT + 0.015],
                   size=(0.03, 0.03, 0.03), color="#2980b9", label="BLUE")
    sim.add_object("green_block", [0.0, -0.1, TABLE_HEIGHT + 0.015],
                   size=(0.03, 0.03, 0.03), color="#27ae60", label="GREEN")

    # 목표 박스 (오른쪽 뒤)
    sim.add_object("box", [0.2, 0.2, TABLE_HEIGHT + 0.025],
                   size=(0.08, 0.08, 0.05), color="#f39c12", label="BOX",
                   is_container=True)


def run_text_mode(lang):
    """텍스트 입력 모드"""
    sim = PickPlaceSimulator()
    parser = CommandParser()
    setup_scene(sim)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Koch v1.1 Voice Pick & Place")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=135)

    plt.ion()
    plt.show()

    current_status = ["대기 중"]
    current_cmd = [""]

    def draw(phase=""):
        if phase:
            current_status[0] = phase
        sim.render(ax, current_status[0], current_cmd[0])
        fig.canvas.draw()
        fig.canvas.flush_events()

    # 초기 렌더링
    draw("명령을 입력하세요")

    print("\n" + "=" * 60)
    print("  Koch v1.1 음성/텍스트 Pick & Place 시뮬레이션")
    print("=" * 60)
    print("\n  테이블 위 물체: RED(빨강), BLUE(파랑), GREEN(초록)")
    print("  목표 컨테이너: BOX (노란 박스)")
    print("\n  명령 예시:")
    print('    "빨간 블록을 박스에 넣어"')
    print('    "파란 물체를 상자에 옮겨"')
    print('    "pick up the green block and put it in the box"')
    print('    "초록 블록 박스"  (간단하게도 가능)')
    print("\n  종료: quit 또는 Ctrl+C")
    print("=" * 60)

    while True:
        try:
            cmd = input("\n명령> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd.lower() in ["quit", "exit", "종료", "q"]:
            break
        if not cmd:
            continue

        current_cmd[0] = f'"{cmd}"'
        draw("명령 분석 중...")

        result = parser.parse(cmd, sim)

        if result is None:
            current_status[0] = "명령을 이해하지 못했습니다"
            current_cmd[0] = f'"{cmd}" (인식 실패)'
            draw()
            print("  -> 명령을 이해하지 못했습니다. 색상 + 박스 키워드를 포함해 주세요.")
            continue

        pick_obj, place_obj = result
        print(f"  -> 잡기: {pick_obj.label} | 놓기: {place_obj.label}")
        current_cmd[0] = f'"{cmd}"  [{pick_obj.label} -> {place_obj.label}]'

        sim.execute_pick_and_place(pick_obj, place_obj, draw_callback=draw)

        # 결과 확인
        dist = np.linalg.norm(pick_obj.position[:2] - place_obj.position[:2])
        if dist < 0.06:
            current_status[0] = f"성공! {pick_obj.label} -> {place_obj.label}"
            print(f"  -> 성공! (거리: {dist*100:.1f}cm)")
        else:
            current_status[0] = f"완료 (거리: {dist*100:.1f}cm)"
            print(f"  -> 완료 (거리: {dist*100:.1f}cm)")

        draw()
        sim.go_home()

    plt.ioff()
    plt.close()
    print("\n종료.")


def run_voice_mode(lang):
    """음성 인식 모드"""
    voice = VoiceInput(lang=lang)
    sim = PickPlaceSimulator()
    parser = CommandParser()
    setup_scene(sim)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Koch v1.1 Voice Pick & Place")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=135)

    plt.ion()
    plt.show()

    current_status = ["대기 중"]
    current_cmd = [""]

    def draw(phase=""):
        if phase:
            current_status[0] = phase
        sim.render(ax, current_status[0], current_cmd[0])
        fig.canvas.draw()
        fig.canvas.flush_events()

    draw("말씀하세요...")

    print("\n" + "=" * 60)
    print("  Koch v1.1 음성 Pick & Place 시뮬레이션")
    print("=" * 60)
    print(f"  언어: {lang}")
    print("  테이블 위 물체: RED, BLUE, GREEN")
    print("  목표: BOX (노란 박스)")
    print('  예: "빨간 블록을 박스에 넣어"')
    print("  종료: Ctrl+C")
    print("=" * 60)

    while True:
        try:
            # 음성 인식 시도
            current_status[0] = "듣는 중..."
            draw()

            text = voice.listen(timeout=10)

            # 음성 실패 시 키보드 폴백
            if text is None:
                if not voice.available:
                    text = input("  (키보드 입력) 명령> ").strip()
                else:
                    print("  (다시 시도하세요)")
                    continue

            if text.lower() in ["quit", "exit", "종료"]:
                break

            current_cmd[0] = f'"{text}"'
            draw("명령 분석 중...")

            result = parser.parse(text, sim)

            if result is None:
                current_status[0] = "명령을 이해하지 못했습니다"
                draw()
                print("  -> 다시 말씀해 주세요.")
                continue

            pick_obj, place_obj = result
            print(f"  -> {pick_obj.label} -> {place_obj.label}")
            current_cmd[0] = f'"{text}"  [{pick_obj.label} -> {place_obj.label}]'

            sim.execute_pick_and_place(pick_obj, place_obj, draw_callback=draw)

            dist = np.linalg.norm(pick_obj.position[:2] - place_obj.position[:2])
            current_status[0] = f"완료! {pick_obj.label} -> {place_obj.label}"
            draw()
            sim.go_home()

        except KeyboardInterrupt:
            break

    plt.ioff()
    plt.close()
    print("\n종료.")


def run_auto_mode(lang):
    """자동 데모 모드 (사전 정의 명령)"""
    sim = PickPlaceSimulator()
    parser = CommandParser()
    setup_scene(sim)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Koch v1.1 Voice Pick & Place (Auto Demo)")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=135)

    plt.ion()
    plt.show()

    current_status = ["자동 데모 시작"]
    current_cmd = [""]

    def draw(phase=""):
        if phase:
            current_status[0] = phase
        sim.render(ax, current_status[0], current_cmd[0])
        fig.canvas.draw()
        fig.canvas.flush_events()

    draw()

    # 자동 명령 시퀀스
    commands = [
        "빨간 블록을 박스에 넣어",
        "파란 블록을 박스에 넣어",
        "초록 블록을 상자에 옮겨",
    ]

    print("\n" + "=" * 60)
    print("  Koch v1.1 자동 Pick & Place 데모")
    print("=" * 60)
    print(f"  {len(commands)}개 명령을 순차 실행합니다\n")

    success = 0
    for idx, cmd in enumerate(commands):
        print(f"\n--- [{idx+1}/{len(commands)}] \"{cmd}\" ---")
        current_cmd[0] = f'[{idx+1}/{len(commands)}] "{cmd}"'
        draw("명령 분석 중...")
        time.sleep(0.5)

        result = parser.parse(cmd, sim)
        if result is None:
            print("  -> 파싱 실패")
            continue

        pick_obj, place_obj = result
        print(f"  -> {pick_obj.label} -> {place_obj.label}")

        sim.execute_pick_and_place(pick_obj, place_obj, draw_callback=draw)

        dist = np.linalg.norm(pick_obj.position[:2] - place_obj.position[:2])
        ok = dist < 0.06
        success += ok
        sym = "SUCCESS" if ok else "RETRY"
        current_status[0] = f"[{idx+1}/{len(commands)}] {pick_obj.label} -> {place_obj.label}: {sym}"
        draw()
        print(f"  -> 거리: {dist*100:.1f}cm [{sym}]")

        sim.go_home()
        time.sleep(0.3)

    current_cmd[0] = f"완료! {success}/{len(commands)} 성공"
    current_status[0] = "자동 데모 종료"
    draw()

    print(f"\n[결과] {success}/{len(commands)} 성공")

    plt.ioff()
    print("\n창을 닫으면 종료됩니다.")
    plt.show()


# ================================================================
# 메인
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Koch v1.1 음성/텍스트 Pick & Place 시뮬레이션",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
모드:
  text     키보드 텍스트 입력 (기본, 추가 패키지 불필요)
  voice    마이크 음성 인식 (speech_recognition + pyaudio 필요)
  auto     사전 정의된 3개 명령 자동 실행 (데모용)

명령 예시:
  "빨간 블록을 박스에 넣어"
  "파란 물체를 상자에 옮겨"
  "pick up the green block and put it in the box"
  "초록 블록 박스"

필요 패키지:
  기본:   pip install numpy matplotlib
  음성:   pip install SpeechRecognition pyaudio
        """,
    )
    parser.add_argument("--mode", default="text",
                        choices=["text", "voice", "auto"])
    parser.add_argument("--lang", default="ko-KR",
                        help="음성 인식 언어 (기본: ko-KR, 영어: en-US)")
    args = parser.parse_args()

    if args.mode == "text":
        run_text_mode(args.lang)
    elif args.mode == "voice":
        run_voice_mode(args.lang)
    elif args.mode == "auto":
        run_auto_mode(args.lang)


if __name__ == "__main__":
    main()
