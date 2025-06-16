# %%
#import libraries

import cv2
import mediapipe as mp
import math

# 初始化 MediaPipe 的臉部偵測
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# 嘴巴關鍵點 index
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# 判斷是否為大笑的閾值（嘴巴張開距離 / 嘴巴寬度）
LAUGH_RATIO_THRESHOLD = 0.5

# 計算兩點之間的歐式距離
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# 開啟攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 翻轉畫面（鏡像）
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 取出口、兩側的點
            top = face_landmarks.landmark[MOUTH_TOP]
            bottom = face_landmarks.landmark[MOUTH_BOTTOM]
            left = face_landmarks.landmark[MOUTH_LEFT]
            right = face_landmarks.landmark[MOUTH_RIGHT]

            # 轉換成 pixel 座標
            top_pt = (int(top.x * w), int(top.y * h))
            bottom_pt = (int(bottom.x * w), int(bottom.y * h))
            left_pt = (int(left.x * w), int(left.y * h))
            right_pt = (int(right.x * w), int(right.y * h))

            # 計算嘴巴張開與寬度
            mouth_open = euclidean_distance(top_pt, bottom_pt)
            mouth_width = euclidean_distance(left_pt, right_pt)
            ratio = mouth_open / mouth_width if mouth_width > 0 else 0

            # 顯示結果
            if ratio > LAUGH_RATIO_THRESHOLD:
                cv2.putText(frame, "Laughing", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Not Laughing", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 顯示嘴巴位置（可選）
            cv2.circle(frame, top_pt, 3, (255, 0, 0), -1)
            cv2.circle(frame, bottom_pt, 3, (255, 0, 0), -1)
            cv2.circle(frame, left_pt, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_pt, 3, (0, 255, 0), -1)

    cv2.imshow("Laugh Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 鍵結束
        break

cap.release()
cv2.destroyAllWindows()



