# ai_interview_coach.py - AI 기반 면접 코치 시스템
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import deque
import math
from queue import Queue, Empty
from datetime import datetime
import json
import dotenv
from openai_advisor import InterviewAdvisor
import os

advisor = None

dotenv.load_dotenv('.env.keys', override=True)

app = Flask(__name__)

def init_ai_advisor():
    """AI 조언 시스템 초기화"""
    global advisor
    try:
        advisor = InterviewAdvisor(
            api_key=os.getenv('AZURE_OPENAI_KEY'),        # 실제 키 이름
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            deployment_name=os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')  # 실제 키 이름
        )
        print("🤖 AI 조언 시스템 초기화 완료")
        return True
    except Exception as e:
        print(f"❌ AI 조언 시스템 초기화 실패: {e}")
        return False

# Socket.IO 최적화 설정
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=30,
    ping_interval=10,
    logger=True,
    engineio_logger=True
)

# MediaPipe 초기화 (성능 최적화)
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 최소 복잡도
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# 손 제스쳐는 선택적 활성화 (성능 고려)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # 임계값 높여서 연산 감소
    min_tracking_confidence=0.7
)

# 큐 크기 최적화
frame_queue = Queue(maxsize=2)  # 3→2로 축소
metrics_queue = Queue(maxsize=5)  # 10→5로 축소

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dx_prev = 0.0
        self.x_prev = None
        self.timestamp_prev = None

    def filter(self, x, timestamp):
        if self.x_prev is None:
            self.x_prev = x
            self.timestamp_prev = timestamp
            return x

        dt = timestamp - self.timestamp_prev
        if dt <= 0:
            return x

        dx = (x - self.x_prev) / dt
        cutoff = self.min_cutoff + self.beta * abs(dx)
        alpha = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * cutoff * dt)))
        x_filtered = alpha * x + (1.0 - alpha) * self.x_prev
        
        self.x_prev = x_filtered
        self.dx_prev = dx
        self.timestamp_prev = timestamp
        
        return x_filtered

# 헤드 포즈 필터
pitch_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)
yaw_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)
roll_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)

class InterviewMetrics:
    def __init__(self):
        # ================================
        # 🎯 조정 가능한 임계값들 (수정금지 - 최적화됨)
        # ================================
        
        # 👁 눈 깜빡임 관련 임계값
        self.ear_threshold = 0.12           # EAR 임계값 (낮을수록 민감, 0.15~0.25 권장)
        self.ear_consecutive_frames = 1     # 깜빡임 인정을 위한 연속 프레임 수 (1~3 권장)
        self.ear_calibration_frames = 30    # 개인별 보정을 위한 프레임 수 (20~50 권장)
        
        # 🎯 머리 끄덕임/흔들기 관련 임계값
        self.nod_threshold = 15             # 끄덕임 감지 각도 임계값 (15~25도 권장)
        self.shake_threshold = 25           # 좌우 흔들기 감지 각도 임계값 (20~35도 권장)
        self.nod_cooldown = 1.0            # 끄덕임 감지 후 대기시간(초) (0.5~2.0 권장)
        self.shake_cooldown = 1.0          # 흔들기 감지 후 대기시간(초) (0.5~2.0 권장)
        self.head_min_frames = 8           # 머리 움직임 판단 최소 프레임 수 (5~15 권장)
        self.nod_pattern_frames = 4        # 끄덕임 패턴 분석 프레임 수 (3~6 권장)
        
        # 📱 자세 흔들림 관련 임계값  
        self.movement_threshold = 0.06      # 어깨 움직임 임계값 (낮을수록 민감, 0.02~0.1 권장)
        self.posture_min_frames = 10       # 자세 판단 최소 프레임 수 (5~20 권장)
        self.posture_sway_cooldown = 2.0   # 자세 흔들림 감지 후 대기시간(초) (1~5 권장)
        
        # 😊 미소 관련 임계값
        self.smile_threshold = 5           # 미소 인정 임계값 (3~10 권장)
        self.smile_intensity_high = 10     # 강한 미소 임계값 (8~15 권장)
        
        # 👋 손 제스쳐 관련 임계값 (성능 최적화됨)
        self.hand_gesture_threshold = 0.03  # 손 움직임 임계값 (약간 높게 설정)
        self.hand_gesture_cooldown = 1.5   # 제스쳐 감지 후 대기시간 (길게 설정)
        self.hand_min_frames = 6           # 제스쳐 판단 최소 프레임 수
        self.hand_analysis_interval = 5    # 5프레임마다 손 분석 (성능 최적화)
        
        # 🚀 성능 최적화 설정
        self.enable_hand_analysis = True   # 손 분석 활성화/비활성화
        self.frame_skip_base = 2           # 기본 프레임 스킵
        self.frame_skip_with_hands = 3     # 손 분석 시 프레임 스킵
        
        # ================================
        # 🕒 AI 면접 코치 전용 변수들
        # ================================
        
        # 분석 세션 시간 추적
        self.analysis_start_time = None     # 분석 시작 시간
        self.analysis_end_time = None       # 분석 종료 시간
        self.session_duration = 0           # 총 세션 시간
        
        # ================================
        # 시스템 변수들 (건드리지 마세요!)
        # ================================
        
        # 분석 제어
        self.analyzing = False
        self.frame_skip = 0
        self.skip_interval = self.frame_skip_base
        
        # 눈 깜빡임 상태 변수
        self.blink_count = 0
        self.last_blink_time = 0
        self.ear_below = False
        self.ear_closed_frames = 0
        self.ear_history_for_calibration = deque(maxlen=self.ear_calibration_frames)
        self.is_ear_calibrated = False
        self.dynamic_ear_threshold = self.ear_threshold
        
        # 머리 자세 상태 변수
        self.head_pose_history = deque(maxlen=15)
        self.nod_count = 0
        self.shake_count = 0
        self.last_nod_time = 0
        self.last_shake_time = 0
        self.pitch_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=5)
        
        # 미세 표정 상태 변수
        self.emotion_history = deque(maxlen=5)
        self.micro_expression_count = 0
        
        # 미소 상태 변수
        self.smile_start_time = None
        self.smile_duration = 0
        self.total_smile_time = 0
        self.smile_intensity_history = deque(maxlen=30)
        
        # 자세 상태 변수
        self.shoulder_positions = deque(maxlen=30)
        self.posture_sway_count = 0
        self.last_sway_time = 0
        
        # 손 제스쳐 상태 변수 (최적화됨)
        self.hand_positions = deque(maxlen=15)  # 크기 축소
        self.hand_gesture_count = 0
        self.last_gesture_time = 0
        self.hands_visible_count = 0
        self.hand_frame_counter = 0  # 손 분석 주기 카운터
        
        # 통계 변수
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_active = True

metrics = InterviewMetrics()

def calibrate_ear_threshold():
    """동적 EAR 임계값 보정"""
    global metrics
    
    if len(metrics.ear_history_for_calibration) >= metrics.ear_calibration_frames:
        ear_values = list(metrics.ear_history_for_calibration)
        mean_ear = np.mean(ear_values)
        std_ear = np.std(ear_values)
        
        metrics.dynamic_ear_threshold = max(0.15, mean_ear - 2*std_ear)
        metrics.is_ear_calibrated = True
        
        print(f"🎯 EAR 임계값 보정 완료: {metrics.dynamic_ear_threshold:.3f}")

def calculate_ear(landmarks, eye_indices):
    """눈 종횡비(EAR) 계산"""
    try:
        coords = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
        A = np.linalg.norm(coords[1] - coords[5])
        B = np.linalg.norm(coords[2] - coords[4])
        C = np.linalg.norm(coords[0] - coords[3])
        return (A + B) / (2.0 * C) if C > 0 else 0
    except (IndexError, ZeroDivisionError):
        return 0

def calculate_head_pose(landmarks, img_size):
    """머리 자세 각도 계산 - One Euro Filter 적용"""
    try:
        h, w = img_size
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],
            [landmarks[175].x * w, landmarks[175].y * h],
            [landmarks[33].x * w, landmarks[33].y * h],
            [landmarks[263].x * w, landmarks[263].y * h],
            [landmarks[61].x * w, landmarks[61].y * h],
            [landmarks[291].x * w, landmarks[291].y * h]
        ], dtype=np.float64)
        
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w/2],
            [0, focal_length, h/2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = math.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
            
            if sy > 1e-6:
                x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = 0
            
            pitch, yaw, roll = math.degrees(x), math.degrees(y), math.degrees(z)
            
            # One Euro Filter 적용
            current_time = time.time()
            filtered_pitch = pitch_filter.filter(pitch, current_time)
            filtered_yaw = yaw_filter.filter(yaw, current_time)
            filtered_roll = roll_filter.filter(roll, current_time)
            
            return filtered_pitch, filtered_yaw, filtered_roll
            
    except Exception as e:
        print(f"Head pose calculation error: {e}")
        return 0, 0, 0

def is_relaxed_nod_pattern(pitch_sequence):
    """완화된 끄덕임 패턴 검증"""
    if len(pitch_sequence) < metrics.nod_pattern_frames:
        return False
    
    pitch_range = max(pitch_sequence) - min(pitch_sequence)
    if pitch_range < 12:
        return False
    
    direction_changes = 0
    for i in range(1, len(pitch_sequence)):
        if i > 1 and (pitch_sequence[i] - pitch_sequence[i-1]) * (pitch_sequence[i-1] - pitch_sequence[i-2]) < 0:
            direction_changes += 1
    
    return direction_changes >= 1

def is_relaxed_shake_pattern(yaw_sequence):
    """완화된 흔들기 패턴 검증"""
    if len(yaw_sequence) < metrics.nod_pattern_frames:
        return False
    
    yaw_range = max(yaw_sequence) - min(yaw_sequence)
    if yaw_range < 15:
        return False
    
    direction_changes = 0
    for i in range(2, len(yaw_sequence)):
        if (yaw_sequence[i] - yaw_sequence[i-1]) * (yaw_sequence[i-1] - yaw_sequence[i-2]) < 0:
            direction_changes += 1
    
    return direction_changes >= 1

def detect_nod_shake(current_pitch, current_yaw):
    """개선된 끄덕임/흔들기 감지"""
    global metrics
    
    current_time = time.time()
    
    metrics.pitch_history.append(current_pitch)
    metrics.yaw_history.append(current_yaw)
    
    if len(metrics.pitch_history) < 5:
        return
    
    # 끄덕임 감지
    if current_time - metrics.last_nod_time > metrics.nod_cooldown:
        if len(metrics.head_pose_history) > metrics.head_min_frames:
            recent_pitches = [pose[0] for pose in list(metrics.head_pose_history)[-metrics.head_min_frames:]]
            pitch_range = max(recent_pitches) - min(recent_pitches)
            
            if pitch_range > metrics.nod_threshold:
                pitch_seq = recent_pitches[-metrics.nod_pattern_frames:]
                if is_relaxed_nod_pattern(pitch_seq):
                    metrics.nod_count += 1
                    metrics.last_nod_time = current_time
                    print(f"✅ 끄덕임 감지! {metrics.nod_count}회")
    
    # 좌우 흔들기 감지
    if current_time - metrics.last_shake_time > metrics.shake_cooldown:
        if len(metrics.head_pose_history) > metrics.head_min_frames:
            recent_yaws = [pose[1] for pose in list(metrics.head_pose_history)[-metrics.head_min_frames:]]
            yaw_range = max(recent_yaws) - min(recent_yaws)
            
            if yaw_range > metrics.shake_threshold:
                yaw_seq = recent_yaws[-metrics.nod_pattern_frames:]
                if is_relaxed_shake_pattern(yaw_seq):
                    metrics.shake_count += 1
                    metrics.last_shake_time = current_time
                    print(f"✅ 좌우 흔들기 감지! {metrics.shake_count}회")

def calculate_smile_intensity(landmarks):
    """미소 강도 계산"""
    try:
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        mouth_center = landmarks[13]
        
        left_lift = mouth_center.y - left_mouth.y
        right_lift = mouth_center.y - right_mouth.y
        
        smile_intensity = (left_lift + right_lift) / 2
        return max(0, smile_intensity * 1000)
    except (IndexError, AttributeError):
        return 0

def improved_posture_sway_detection(shoulder_landmarks):
    """개선된 자세 흔들림 감지"""
    global metrics
    
    current_time = time.time()
    
    if current_time - metrics.last_sway_time < metrics.posture_sway_cooldown:
        return
    
    if not shoulder_landmarks:
        return
    
    left_shoulder = shoulder_landmarks.get(11)
    right_shoulder = shoulder_landmarks.get(12)
    
    if left_shoulder and right_shoulder:
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        metrics.shoulder_positions.append((center_x, center_y))
        
        if len(metrics.shoulder_positions) > metrics.posture_min_frames:
            recent_positions = list(metrics.shoulder_positions)[-metrics.posture_min_frames:]
            x_positions = [pos[0] for pos in recent_positions]
            y_positions = [pos[1] for pos in recent_positions]
            
            x_range = max(x_positions) - min(x_positions)
            y_range = max(y_positions) - min(y_positions)
            
            if x_range > metrics.movement_threshold or y_range > metrics.movement_threshold:
                metrics.posture_sway_count += 1
                metrics.last_sway_time = current_time
                print(f"📱 자세 흔들림 감지! {metrics.posture_sway_count}회")

def detect_hand_gestures_optimized(hand_results):
    """최적화된 손 제스쳐 감지 (주기적 실행)"""
    global metrics
    
    # 성능 최적화: N프레임마다만 실행
    metrics.hand_frame_counter += 1
    if metrics.hand_frame_counter < metrics.hand_analysis_interval:
        if hand_results.multi_hand_landmarks:
            metrics.hands_visible_count += 1
        return
    
    metrics.hand_frame_counter = 0
    current_time = time.time()
    
    if hand_results.multi_hand_landmarks:
        metrics.hands_visible_count += 1
        
        # 간단한 손목 위치만 추적 (성능 최적화)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            metrics.hand_positions.append((wrist.x, wrist.y, current_time))
        
        # 제스쳐 감지 (쿨다운 적용)
        if current_time - metrics.last_gesture_time > metrics.hand_gesture_cooldown:
            if len(metrics.hand_positions) > metrics.hand_min_frames:
                recent_positions = list(metrics.hand_positions)[-metrics.hand_min_frames:]
                
                if len(recent_positions) > 1:
                    x_positions = [pos[0] for pos in recent_positions]
                    y_positions = [pos[1] for pos in recent_positions]
                    
                    x_range = max(x_positions) - min(x_positions)
                    y_range = max(y_positions) - min(y_positions)
                    
                    if x_range > metrics.hand_gesture_threshold or y_range > metrics.hand_gesture_threshold:
                        metrics.hand_gesture_count += 1
                        metrics.last_gesture_time = current_time
                        print(f"👋 손 제스쳐 감지! {metrics.hand_gesture_count}회")

def get_detailed_analysis_data():
    """AI 분석을 위한 상세 데이터 생성"""
    global metrics
    
    # 세션 시간 계산
    if metrics.analysis_start_time:
        if metrics.analysis_end_time:
            session_duration = metrics.analysis_end_time - metrics.analysis_start_time
        else:
            session_duration = time.time() - metrics.analysis_start_time
    else:
        session_duration = 0
    
    # 상세 분석 데이터
    analysis_data = {
        'session_info': {
            'duration_seconds': round(session_duration, 1),
            'total_frames': metrics.frame_count,
            'analysis_timestamp': datetime.now().isoformat(),
            'start_time': datetime.fromtimestamp(metrics.analysis_start_time).isoformat() if metrics.analysis_start_time else None,
            'end_time': datetime.fromtimestamp(metrics.analysis_end_time).isoformat() if metrics.analysis_end_time else None
        },
        'behavioral_metrics': {
            'eye_contact': {
                'blink_count': metrics.blink_count,
                'blink_rate_per_minute': round(metrics.blink_count / (session_duration / 60), 1) if session_duration > 0 else 0,
                'average_ear': round(np.mean(list(metrics.ear_history_for_calibration)) if metrics.ear_history_for_calibration else 0, 3)
            },
            'facial_expressions': {
                'total_smile_time': round(metrics.total_smile_time, 1),
                'smile_percentage': round((metrics.total_smile_time / session_duration) * 100, 1) if session_duration > 0 else 0,
                'average_smile_intensity': round(np.mean(list(metrics.smile_intensity_history)) if metrics.smile_intensity_history else 0, 2),
                'micro_expressions': metrics.micro_expression_count
            },
            'head_movements': {
                'nod_count': metrics.nod_count,
                'shake_count': metrics.shake_count,
                'head_stability_score': calculate_head_stability()
            },
            'posture': {
                'sway_count': metrics.posture_sway_count,
                'stability_score': calculate_posture_stability()
            },
            'hand_gestures': {
                'gesture_count': metrics.hand_gesture_count,
                'hands_visible_seconds': round((metrics.hands_visible_count / 15), 1) if metrics.frame_count > 0 else 0,
                'gesture_frequency_per_minute': round(metrics.hand_gesture_count / (session_duration / 60), 1) if session_duration > 0 else 0
            }
        },
        'scientific_standards': {
            'normal_blink_rate_range': {'min': 12, 'max': 20, 'unit': 'per_minute'},
            'optimal_smile_percentage_range': {'min': 25, 'max': 40, 'unit': 'percent'},
            'appropriate_gesture_frequency_range': {'min': 2, 'max': 8, 'unit': 'per_minute'},
            'head_movement_balance': {
                'minimal_nods': 3,
                'optimal_nods': 8, 
                'excessive_nods': 15,
                'excessive_shake': 5
            }
        }
    }
    
    return analysis_data

def calculate_head_stability():
    """머리 안정성 점수 계산 (0-100)"""
    if not metrics.head_pose_history:
        return 100
    
    recent_poses = list(metrics.head_pose_history)[-30:]
    if len(recent_poses) < 10:
        return 100
    
    pitches = [pose[0] for pose in recent_poses]
    yaws = [pose[1] for pose in recent_poses]
    
    pitch_std = np.std(pitches)
    yaw_std = np.std(yaws)
    
    stability = max(0, 100 - (pitch_std + yaw_std) * 2)
    return round(stability, 1)

def calculate_posture_stability():
    """자세 안정성 점수 계산 (0-100)"""
    if not metrics.shoulder_positions:
        return 100
    
    positions = list(metrics.shoulder_positions)[-50:]
    if len(positions) < 20:
        return 100
    
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)
    
    stability = max(0, 100 - (x_std + y_std) * 1000)
    return round(stability, 1)

# 기존 프레임 처리 로직 (process_frame 함수 등) - 동일하게 유지
def process_frame(frame):
    """프레임 처리 및 지표 계산 - 성능 최적화됨"""
    global metrics
    
    if metrics.enable_hand_analysis:
        metrics.skip_interval = metrics.frame_skip_with_hands
    else:
        metrics.skip_interval = metrics.frame_skip_base
    
    metrics.frame_skip += 1
    if metrics.frame_skip < metrics.skip_interval:
        return {}, frame
    
    metrics.frame_skip = 0
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        current_metrics = {
            'timestamp': time.time(),
            'blink_count': metrics.blink_count,
            'ear': 0.0,
            'head_pose': {'pitch': 0, 'yaw': 0, 'roll': 0},
            'nod_count': metrics.nod_count,
            'shake_count': metrics.shake_count,
            'emotion': 'neutral',
            'micro_expressions': metrics.micro_expression_count,
            'smile_intensity': 0,
            'smile_duration': metrics.smile_duration,
            'total_smile_time': metrics.total_smile_time,
            'posture_sway': metrics.posture_sway_count,
            'hand_gesture_count': metrics.hand_gesture_count,
            'hands_visible_count': metrics.hands_visible_count,
            'frame_count': metrics.frame_count
        }
        
        face_results = face_mesh.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        hand_results = None
        
        if metrics.enable_hand_analysis:
            hand_results = hands.process(rgb_frame)
        
        if face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )
            
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            left_ear = calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])
            ear = (left_ear + right_ear) / 2
            current_metrics['ear'] = round(ear, 3)
            
            if not metrics.is_ear_calibrated:
                metrics.ear_history_for_calibration.append(ear)
                if len(metrics.ear_history_for_calibration) >= metrics.ear_calibration_frames:
                    calibrate_ear_threshold()
            
            if metrics.analyzing:
                current_threshold = metrics.dynamic_ear_threshold if metrics.is_ear_calibrated else metrics.ear_threshold
                
                if ear < current_threshold:
                    metrics.ear_closed_frames += 1
                    if not metrics.ear_below and metrics.ear_closed_frames >= metrics.ear_consecutive_frames:
                        metrics.ear_below = True
                else:
                    if metrics.ear_below and metrics.ear_closed_frames >= metrics.ear_consecutive_frames:
                        metrics.blink_count += 1
                        metrics.last_blink_time = time.time()
                        current_metrics['blink_count'] = metrics.blink_count
                        print(f"✅ 깜빡임 #{metrics.blink_count}번!")
                    
                    metrics.ear_below = False
                    metrics.ear_closed_frames = 0
                
                pitch, yaw, roll = calculate_head_pose(landmarks, (h, w))
                current_metrics['head_pose'] = {
                    'pitch': round(pitch, 1),
                    'yaw': round(yaw, 1),
                    'roll': round(roll, 1)
                }
                
                metrics.head_pose_history.append((pitch, yaw, roll))
                detect_nod_shake(pitch, yaw)
                current_metrics['nod_count'] = metrics.nod_count
                current_metrics['shake_count'] = metrics.shake_count
                
                smile_intensity = calculate_smile_intensity(landmarks)
                current_metrics['smile_intensity'] = round(smile_intensity, 2)
                metrics.smile_intensity_history.append(smile_intensity)
                
                if smile_intensity > metrics.smile_threshold:
                    if metrics.smile_start_time is None:
                        metrics.smile_start_time = time.time()
                    else:
                        metrics.smile_duration = time.time() - metrics.smile_start_time
                else:
                    if metrics.smile_start_time is not None:
                        metrics.total_smile_time += metrics.smile_duration
                        metrics.smile_start_time = None
                        metrics.smile_duration = 0
                
                current_metrics['smile_duration'] = round(metrics.smile_duration, 1)
                current_metrics['total_smile_time'] = round(metrics.total_smile_time, 1)
                
                if smile_intensity > metrics.smile_intensity_high:
                    emotion = 'happy'
                elif abs(pitch) > 20 or abs(yaw) > 30:
                    emotion = 'surprised'
                else:
                    emotion = 'neutral'
                
                current_metrics['emotion'] = emotion
                
                if len(metrics.emotion_history) > 0:
                    if metrics.emotion_history[-1] != emotion:
                        metrics.micro_expression_count += 1
                
                metrics.emotion_history.append(emotion)
                current_metrics['micro_expressions'] = metrics.micro_expression_count
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
            )
            
            if metrics.analyzing:
                pose_landmarks = {}
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    pose_landmarks[idx] = landmark
                
                improved_posture_sway_detection(pose_landmarks)
                current_metrics['posture_sway'] = metrics.posture_sway_count
        
        if metrics.enable_hand_analysis and hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,255), thickness=2)
                )
            
            if metrics.analyzing:
                detect_hand_gestures_optimized(hand_results)
                current_metrics['hand_gesture_count'] = metrics.hand_gesture_count
                current_metrics['hands_visible_count'] = metrics.hands_visible_count
        
        if metrics.analyzing:
            metrics.frame_count += 1
            current_metrics['frame_count'] = metrics.frame_count
        
        return current_metrics, frame
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        return {}, frame

def process_frame_worker():
    """별도 스레드에서 프레임 처리"""
    while metrics.processing_active:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            
            current_metrics, processed_frame = process_frame(frame)
            
            if not metrics_queue.full():
                metrics_queue.put(current_metrics)
            
            frame_queue.task_done()
            
        except Empty:
            continue
        except Exception as e:
            print(f"Frame processing worker error: {e}")
            time.sleep(0.1)

def metrics_sender_worker():
    """메트릭을 Socket.IO로 전송하는 별도 스레드"""
    while metrics.processing_active:
        try:
            current_metrics = metrics_queue.get(timeout=1)
            if current_metrics:
                socketio.emit('metrics_update', current_metrics)
                metrics_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"Metrics sender error: {e}")
            time.sleep(0.1)

# 비디오 스트리밍 최적화
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 15)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate_frames():
    processing_thread = threading.Thread(target=process_frame_worker, daemon=True)
    processing_thread.start()
    
    metrics_thread = threading.Thread(target=metrics_sender_worker, daemon=True)
    metrics_thread.start()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if frame_queue.full():
            try:
                _ = frame_queue.get_nowait()
            except Empty:
                pass
        
        frame_queue.put(frame.copy())
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('interview_coach.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================
# 🤖 AI 면접 코치 전용 API
# ================================

@app.route('/api/get_analysis_data')
def get_analysis_data():
    """AI 분석을 위한 상세 데이터 반환"""
    try:
        analysis_data = get_detailed_analysis_data()
        return {
            'status': 'success',
            'data': analysis_data
        }
    except Exception as e:
        print(f"Get analysis data error: {e}")
        return {'status': 'error', 'message': str(e)}

# Socket.IO 이벤트 핸들러들
@socketio.on_error()
def error_handler(e):
    print(f"❌ SocketIO Error: {e}")
    return True

@socketio.on('connect')
def handle_connect():
    print(f"🌐 클라이언트 연결됨: {request.sid}")
    emit('connection_status', {'status': 'connected', 'timestamp': time.time()})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🌐 클라이언트 연결 끊김: {request.sid}")

@socketio.on('toggle_analysis')
def toggle_analysis(data):
    """분석 시작/중지 토글 + 시간 추적"""
    try:
        print(f"🚦 서버: TOGGLE_RECEIVED {data}")
        analyzing = data.get('analyze', False)
        
        if analyzing:
            # 분석 시작
            metrics.analysis_start_time = time.time()
            metrics.analysis_end_time = None
            print(f"🎯 면접 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            # 분석 종료
            metrics.analysis_end_time = time.time()
            if metrics.analysis_start_time:
                session_duration = metrics.analysis_end_time - metrics.analysis_start_time
                print(f"🎯 면접 분석 완료: 총 {session_duration:.1f}초")
            
        metrics.analyzing = analyzing
        
        emit('analysis_status', {
            'analyzing': metrics.analyzing,
            'timestamp': time.time(),
            'session_duration': session_duration if not analyzing and metrics.analysis_end_time else 0
        })
        
    except Exception as e:
        print(f"Toggle analysis error: {e}")
        emit('error', {'message': f'Analysis toggle failed: {str(e)}'})

@socketio.on('toggle_hand_analysis')
def toggle_hand_analysis(data):
    """손 분석 활성화/비활성화"""
    try:
        metrics.enable_hand_analysis = data.get('enable', True)
        print(f"👋 손 분석 상태 변경: {metrics.enable_hand_analysis}")
        
        emit('hand_analysis_status', {
            'enabled': metrics.enable_hand_analysis,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Toggle hand analysis error: {e}")

@app.route('/api/reset_metrics')
def reset_metrics():
    """메트릭 초기화"""
    try:
        global metrics
        old_analyzing = metrics.analyzing
        old_hand_enabled = metrics.enable_hand_analysis
        metrics = InterviewMetrics()
        metrics.analyzing = old_analyzing
        metrics.enable_hand_analysis = old_hand_enabled
        print("🔄 메트릭 초기화 완료")
        return {'status': 'success', 'message': 'Metrics reset successfully'}
    except Exception as e:
        print(f"Reset metrics error: {e}")
        return {'status': 'error', 'message': str(e)}

@app.route('/api/get_summary')
def get_summary():
    """세션 요약 통계"""
    try:
        elapsed_time = time.time() - metrics.start_time
        
        # 분석 세션 시간 계산
        if metrics.analysis_start_time:
            if metrics.analysis_end_time:
                analysis_session_time = metrics.analysis_end_time - metrics.analysis_start_time
            else:
                analysis_session_time = time.time() - metrics.analysis_start_time
        else:
            analysis_session_time = 0
        
        hands_visible_seconds = round((metrics.hands_visible_count / 15), 1) if metrics.frame_count > 0 else 0
        
        return {
            'status': 'success',
            'data': {
                'session_duration': round(elapsed_time, 1),
                'analysis_session_duration': round(analysis_session_time, 1),  # 실제 분석 시간
                'total_frames': metrics.frame_count,
                'blink_rate': round(metrics.blink_count / (analysis_session_time / 60), 1) if analysis_session_time > 0 else 0,
                'nod_count': metrics.nod_count,
                'shake_count': metrics.shake_count,
                'micro_expressions': metrics.micro_expression_count,
                'total_smile_time': round(metrics.total_smile_time, 1),
                'posture_sway_count': metrics.posture_sway_count,
                'hand_gesture_count': metrics.hand_gesture_count,
                'hands_visible_seconds': hands_visible_seconds,
                'hands_visible_rate': round((metrics.hands_visible_count / metrics.frame_count) * 100, 1) if metrics.frame_count > 0 else 0,
                'avg_smile_intensity': round(np.mean(list(metrics.smile_intensity_history)) if metrics.smile_intensity_history else 0, 2)
            }
        }
    except Exception as e:
        print(f"Get summary error: {e}")
        return {'status': 'error', 'message': str(e)}

@app.route('/api/health')
def health_check():
    """서버 상태 확인"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'analyzing': metrics.analyzing,
        'frame_count': metrics.frame_count,
        'hand_analysis_enabled': metrics.enable_hand_analysis
    }

@app.route('/api/generate_ai_advice', methods=['POST'])
def generate_ai_advice():
    """AI 기반 면접 조언 생성"""
    try:
        global advisor
        if not advisor:
            return {
                'status': 'error',
                'message': 'AI 조언 시스템이 초기화되지 않았습니다.',
                'fallback_advice': '기본 조언을 사용합니다. Azure OpenAI 설정을 확인해주세요.'
            }
        
        analysis_data = request.get_json()
        if not analysis_data:
            return {'status': 'error', 'message': '분석 데이터가 제공되지 않았습니다.'}
        
        print(f"🤖 AI 조언 생성 시작...")
        advice_result = advisor.generate_advice(analysis_data)
        
        if advice_result['status'] == 'success':
            print(f"✅ AI 조언 생성 성공")
            return {
                'status': 'success',
                'advice': advice_result['advice'],
                'analysis_summary': advice_result.get('analysis_summary', {}),
                'timestamp': advice_result['timestamp']
            }
        else:
            print(f"⚠️ AI 조언 생성 실패, 기본 조언 제공")
            return {
                'status': 'error',
                'message': advice_result.get('message', 'AI 조언 생성 실패'),
                'fallback_advice': advice_result.get('fallback_advice', '기본 조언을 생성할 수 없습니다.')
            }
    except Exception as e:
        print(f"❌ AI 조언 API 오류: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'fallback_advice': '서버 오류로 인해 조언을 생성할 수 없습니다.'
        }

if __name__ == '__main__':
    print("🤖 AI 면접 코치 서버 시작중...")
    
    # AI 시스템 초기화
    api_key = os.getenv('AZURE_OPENAI_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if api_key and endpoint:
        print("✅ Azure OpenAI 인증 정보 확인됨")
        init_success = init_ai_advisor()
        if init_success:
            print("🎯 AI 조언 기능 활성화")
        else:
            print("⚠️ AI 조언 기능 비활성화")
    else:
        print("⚠️ Azure OpenAI 인증 정보 없음")

    try:
        print("🚀 Flask 서버 시작...")
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        metrics.processing_active = False
        camera.release()
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        metrics.processing_active = False
        camera.release()