# ============================================================
# hand_collect.py
# Edge AI 시리즈 2편 - 학습 데이터 수집 도구
#
# 1편의 hand.py에 데이터 수집 기능을 추가한 버전입니다.
#  - PanTilt 추적은 그대로 동작
#  - 키보드 단축키로 4가지 클래스 PNG를 클래스별 폴더에 저장
#  - 캡처 직전 segmentation 결과를 미리보기로 표시
#  - segmentation 실패 시 캡처 차단 (잘못된 데이터 방지)
#
# 키보드 단축키:
#   1 = fist
#   2 = index
#   3 = victory
#   4 = ok
#   p = segmentation 미리보기 토글
#   c = 현재 클래스별 수집 개수 출력
#   q = 종료
# ============================================================

import os
import cv2
import mediapipe as mp
import socket

from segmenter import segment_hand_to_96x96, to_grayscale


# ============================================================
# 설정 - 본인 환경에 맞게 수정
# ============================================================
ESP32_IP = "192.168.219.xxx"
ESP32_PORT = 4210
URL = "http://192.168.219.xxx:81/stream"

# 데이터 저장 루트 폴더
DATASET_ROOT = r"C:\dataset"

# 클래스명과 키보드 매핑
CLASS_KEYS = {
    ord('1'): "fist",
    ord('2'): "index",
    ord('3'): "victory",
    ord('4'): "ok",
}

# 저장 형식
SAVE_AS_GRAYSCALE = True   # True: grayscale PNG, False: BGR PNG


# ============================================================
# 폴더 준비 - 클래스별 디렉토리 생성
# ============================================================
def prepare_dirs(root, classes):
    """클래스별 폴더가 없으면 생성하고, 현재 수집 개수를 반환"""
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        # 기존 파일 개수 확인 (이어서 수집할 수 있도록)
        existing = [f for f in os.listdir(cls_dir) if f.lower().endswith('.png')]
        counts[cls] = len(existing)
    return counts


# ============================================================
# 저장 함수
# ============================================================
def save_sample(img_bgr, class_name, current_count):
    """
    segmented 이미지를 클래스 폴더에 저장.
    파일명은 클래스명_순번.png 형식 (예: fist_0001.png).
    """
    if SAVE_AS_GRAYSCALE:
        img_to_save = to_grayscale(img_bgr)
    else:
        img_to_save = img_bgr

    filename = f"{class_name}_{current_count + 1:04d}.png"
    filepath = os.path.join(DATASET_ROOT, class_name, filename)
    cv2.imwrite(filepath, img_to_save)
    return filepath


# ============================================================
# UDP/MediaPipe 초기화 (1편과 동일)
# ============================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(URL)
print("Stream open:", cap.isOpened())


# ============================================================
# 추적 상태 (1편과 동일) - PanTilt 추적을 위한 좌표 송신용
# ============================================================
last_x = -1
last_y = -1
miss_count = 0
MAX_MISS = 20

x_history = []
y_history = []
SMOOTH_N = 8


# ============================================================
# 수집 상태
# ============================================================
class_names = list(CLASS_KEYS.values())
counts = prepare_dirs(DATASET_ROOT, class_names)

print("\n=== 데이터 수집 시작 ===")
print(f"저장 위치: {DATASET_ROOT}")
print(f"저장 형식: {'grayscale' if SAVE_AS_GRAYSCALE else 'BGR'} PNG")
print("\n키보드 단축키:")
print("  1 = fist    2 = index    3 = victory    4 = ok")
print("  p = segmentation 미리보기 토글")
print("  c = 현재 수집 개수 출력")
print("  q = 종료")
print("\n현재 수집 개수:")
for cls in class_names:
    print(f"  {cls:10s}: {counts[cls]:3d} 장")
print()


# ============================================================
# 미리보기 창 토글 상태
# ============================================================
show_preview = True

# 캡처 직전의 segmented 결과 캐싱 (미리보기와 저장 양쪽에서 사용)
# 키 누른 순간 다시 segmentation하면 손이 살짝 움직였을 수 있어서
# 매 프레임 미리 계산해두고 그 결과를 저장
latest_segmented = None
latest_landmarks_valid = False


# ============================================================
# 메인 루프
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    # ----- 손이 검출된 경우 -----
    latest_landmarks_valid = False
    latest_segmented = None

    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            # ===== 1편과 동일 - PanTilt 추적용 좌표 송신 =====
            palm = hl.landmark[9]
            px = int(palm.x * w)
            py = int(palm.y * h)

            x_history.append(px)
            y_history.append(py)
            if len(x_history) > SMOOTH_N:
                x_history.pop(0)
                y_history.pop(0)

            avg_x = int(sum(x_history) / len(x_history))
            avg_y = int(sum(y_history) / len(y_history))

            cv2.circle(frame, (avg_x, avg_y), 15, (0, 0, 255), -1)

            last_x = avg_x
            last_y = avg_y
            miss_count = 0

            msg = "P," + str(avg_x) + "," + str(avg_y)
            sock.sendto(msg.encode(), (ESP32_IP, ESP32_PORT))

            # ===== 2편 신규 - segmentation 미리 계산 =====
            # 키 누르는 순간 다시 호출하지 않고, 매 프레임 미리 계산
            # 키 누른 순간의 결과는 곧 직전 프레임의 결과와 거의 같음
            seg = segment_hand_to_96x96(frame.copy(), hl)
            if seg is not None:
                latest_segmented = seg
                latest_landmarks_valid = True

            # 첫 번째 손만 처리
            break

    else:
        # 손이 없을 때 - 1편과 동일
        miss_count += 1
        if miss_count < MAX_MISS and last_x >= 0:
            msg = "P," + str(last_x) + "," + str(last_y)
            sock.sendto(msg.encode(), (ESP32_IP, ESP32_PORT))
        else:
            sock.sendto(b"N", (ESP32_IP, ESP32_PORT))

    # ----- 화면 표시 -----
    # 좌상단에 수집 진행 상태 오버레이
    y_text = 25
    cv2.putText(frame, "[1]fist [2]index [3]victory [4]ok",
                (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    y_text += 20
    for cls in class_names:
        line = f"{cls}: {counts[cls]:3d}"
        cv2.putText(frame, line, (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if counts[cls] >= 100 else (255, 255, 255), 1)
        y_text += 18

    # segmentation 가능 여부 표시
    if latest_landmarks_valid:
        cv2.putText(frame, "READY", (w - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO HAND", (w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 100, 255), 2)

    cv2.imshow("ESP32-CAM Collect", frame)

    # 미리보기 창 (96x96 segmentation 결과를 확대해서 표시)
    if show_preview and latest_segmented is not None:
        preview = cv2.resize(latest_segmented, (240, 240),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Segmented Preview (240x240 of 96x96)", preview)
    elif show_preview:
        # 손이 없으면 검은 화면
        blank = cv2.resize(
            cv2.cvtColor(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 0,
                cv2.COLOR_GRAY2BGR),
            (240, 240))
        cv2.putText(blank, "no hand", (75, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)
        cv2.imshow("Segmented Preview (240x240 of 96x96)", blank)

    # ----- 키 입력 처리 -----
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        show_preview = not show_preview
        if not show_preview:
            cv2.destroyWindow("Segmented Preview (240x240 of 96x96)")
        print(f"미리보기: {'ON' if show_preview else 'OFF'}")

    elif key == ord('c'):
        print("\n=== 현재 수집 개수 ===")
        for cls in class_names:
            mark = " (목표 달성)" if counts[cls] >= 100 else ""
            print(f"  {cls:10s}: {counts[cls]:3d} 장{mark}")
        print()

    elif key in CLASS_KEYS:
        # 클래스 캡처 키 (1, 2, 3, 4)
        class_name = CLASS_KEYS[key]

        if not latest_landmarks_valid or latest_segmented is None:
            print(f"  [skip] 손이 검출되지 않아 {class_name} 저장 안 함")
        else:
            filepath = save_sample(latest_segmented, class_name, counts[class_name])
            counts[class_name] += 1
            print(f"  [save] {class_name}_{counts[class_name]:04d}.png "
                  f"(누적 {counts[class_name]}장)")


# ============================================================
# 정리
# ============================================================
cap.release()
cv2.destroyAllWindows()
sock.close()

print("\n=== 수집 완료 ===")
for cls in class_names:
    print(f"  {cls:10s}: {counts[cls]:3d} 장")
