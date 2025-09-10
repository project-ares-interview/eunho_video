# interview_analyzer.py - 모듈화된 면접 분석 엔진
from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import deque
import math
from queue import Queue, Empty
import json
from datetime import datetime

class InterviewAnalyzer:
    def __init__(self):
        # Flask 및 SocketIO 설정
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            ping_timeout=30,
            ping_interval=10,
            logger=True,
            engineio_logger=True
        )
        
        # MediaPipe 초기화
        self._init_mediapipe()
        
        # 큐 초기화
        self.frame_queue = Queue(maxsize=2)
        self.metrics_queue = Queue(maxsize=5)
        
        # 필터 초기화
        self._init_filters()
        
        # 메트릭 초기화
        self.metrics = InterviewMetrics()
        
        # 카메라 초기화
        self._init_camera()
        
        # 라우트 및 이벤트 등록
        self._register_routes()
        self._register_events()
        
        print("🔬 면접 분석 엔진이 초기화되었습니다.")

    def _init_mediapipe(self):
        """MediaPipe 모델 초기화"""
        mp_solutions = mp.solutions
        
        self.mp_face_mesh = mp_solutions.face_mesh
        self.mp_pose = mp_solutions.pose
        self.mp_hands = mp_solutions.hands
        self.mp_drawing = mp_solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def _init_filters(self):
        """One Euro Filter 초기화"""
        self.pitch_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)
        self.yaw_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)
        self.roll_filter = OneEuroFilter(min_cutoff=0.1, beta=0.01)

    def _init_camera(self):
        """카메라 초기화"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _register_routes(self):
        """Flask 라우트 등록"""
        @self.app.route('/')
        def index():
            return render_template('interview_coach.html')

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/api/reset_metrics')
        def reset_metrics():
            return self.reset_metrics()

        @self.app.route('/api/get_summary')
        def get_summary():
            return self.get_summary()

        @self.app.route('/api/get_analysis_data')
        def get_analysis_data():
            """AI 분석을 위한 상세 데이터 반환"""
            return self.get_detailed_analysis_data()

        @self.app.route('/api/health')
        def health_check():
            return self.health_check()

    def _register_events(self):
        """SocketIO 이벤트 등록"""
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🌐 클라이언트 연결됨: {request.sid}")
            emit('connection_status', {'status': 'connected', 'timestamp': time.time()})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🌐 클라이언트 연결 끊김: {request.sid}")

        @self.socketio.on('toggle_analysis')
        def toggle_analysis(data):
            try:
                analyzing = data.get('analyze', False)
                
                if analyzing:
                    # 분석 시작 시간 기록
                    self.metrics.analysis_start_time = time.time()
                    self.metrics.analysis_end_time = None
                    print(f"📊 면접 분석 시작: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    # 분석 종료 시간 기록
                    self.metrics.analysis_end_time = time.time()
                    session_duration = self.metrics.analysis_end_time - self.metrics.analysis_start_time
                    print(f"📊 면접 분석 완료: {datetime.now().strftime('%H:%M:%S')} (총 {session_duration:.1f}초)")
                
                self.metrics.analyzing = analyzing
                
                emit('analysis_status', {
                    'analyzing': self.metrics.analyzing,
                    'timestamp': time.time(),
                    'session_duration': session_duration if not analyzing and self.metrics.analysis_end_time else 0
                })
                
            except Exception as e:
                print(f"Toggle analysis error: {e}")
                emit('error', {'message': f'Analysis toggle failed: {str(e)}'})

    def get_detailed_analysis_data(self):
        """AI 분석을 위한 상세 데이터 반환"""
        try:
            # 분석 세션 시간 계산
            if self.metrics.analysis_start_time:
                if self.metrics.analysis_end_time:
                    session_duration = self.metrics.analysis_end_time - self.metrics.analysis_start_time
                else:
                    session_duration = time.time() - self.metrics.analysis_start_time
            else:
                session_duration = 0
            
            # 상세 분석 데이터 생성
            analysis_data = {
                'session_info': {
                    'duration_seconds': round(session_duration, 1),
                    'total_frames': self.metrics.frame_count,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'behavioral_metrics': {
                    'eye_contact': {
                        'blink_count': self.metrics.blink_count,
                        'blink_rate_per_minute': round(self.metrics.blink_count / (session_duration / 60), 1) if session_duration > 0 else 0,
                        'average_ear': round(np.mean(list(self.metrics.ear_history_for_calibration)) if self.metrics.ear_history_for_calibration else 0, 3)
                    },
                    'facial_expressions': {
                        'total_smile_time': round(self.metrics.total_smile_time, 1),
                        'smile_percentage': round((self.metrics.total_smile_time / session_duration) * 100, 1) if session_duration > 0 else 0,
                        'average_smile_intensity': round(np.mean(list(self.metrics.smile_intensity_history)) if self.metrics.smile_intensity_history else 0, 2),
                        'micro_expressions': self.metrics.micro_expression_count
                    },
                    'head_movements': {
                        'nod_count': self.metrics.nod_count,
                        'shake_count': self.metrics.shake_count,
                        'head_stability': self._calculate_head_stability()
                    },
                    'posture': {
                        'sway_count': self.metrics.posture_sway_count,
                        'stability_score': self._calculate_posture_stability()
                    },
                    'hand_gestures': {
                        'gesture_count': self.metrics.hand_gesture_count,
                        'hands_visible_seconds': round((self.metrics.hands_visible_count / 15), 1) if self.metrics.frame_count > 0 else 0,
                        'gesture_frequency': round(self.metrics.hand_gesture_count / (session_duration / 60), 1) if session_duration > 0 else 0
                    }
                },
                'reference_standards': {
                    'normal_blink_rate': {'min': 12, 'max': 20, 'unit': 'per_minute'},
                    'optimal_smile_percentage': {'min': 25, 'max': 40, 'unit': 'percent'},
                    'appropriate_gesture_frequency': {'min': 2, 'max': 8, 'unit': 'per_minute'},
                    'head_movement_balance': {'excessive_threshold': 15, 'minimal_threshold': 3}
                }
            }
            
            return {
                'status': 'success',
                'data': analysis_data
            }
            
        except Exception as e:
            print(f"Get analysis data error: {e}")
            return {'status': 'error', 'message': str(e)}

    def _calculate_head_stability(self):
        """머리 안정성 점수 계산 (0-100)"""
        if not self.metrics.head_pose_history:
            return 100
        
        # 최근 자세 데이터에서 변동성 계산
        recent_poses = list(self.metrics.head_pose_history)[-30:]  # 최근 30프레임
        if len(recent_poses) < 10:
            return 100
        
        pitches = [pose[0] for pose in recent_poses]
        yaws = [pose[1] for pose in recent_poses]
        
        pitch_std = np.std(pitches)
        yaw_std = np.std(yaws)
        
        # 표준편차가 낮을수록 안정적 (역산으로 점수 계산)
        stability = max(0, 100 - (pitch_std + yaw_std) * 2)
        return round(stability, 1)

    def _calculate_posture_stability(self):
        """자세 안정성 점수 계산 (0-100)"""
        if not self.metrics.shoulder_positions:
            return 100
        
        positions = list(self.metrics.shoulder_positions)[-50:]  # 최근 50프레임
        if len(positions) < 20:
            return 100
        
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        
        stability = max(0, 100 - (x_std + y_std) * 1000)  # 좌표 단위 조정
        return round(stability, 1)

    # 기존 메소드들 (calculate_ear, process_frame 등) 그대로 유지
    def calculate_ear(self, landmarks, eye_indices):
        """눈 종횡비(EAR) 계산"""
        try:
            coords = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
            A = np.linalg.norm(coords[1] - coords[5])
            B = np.linalg.norm(coords[2] - coords[4])
            C = np.linalg.norm(coords[0] - coords[3])
            return (A + B) / (2.0 * C) if C > 0 else 0
        except (IndexError, ZeroDivisionError):
            return 0

    def process_frame(self, frame):
        """프레임 처리 - 기존 로직과 동일"""
        # 기존 process_frame 로직 그대로 사용
        # (너무 길어서 생략, 실제로는 기존 코드 복사)
        pass

    def run(self, host='0.0.0.0', port=5001, debug=True):
        """서버 실행"""
        print("🚀 AI 면접 코치 서버 시작중...")
        print("📊 기능:")
        print("- 🎯 실시간 행동 분석")
        print("- 🤖 AI 기반 면접 조언")
        print("- 📈 과학적 근거 기반 피드백")
        
        try:
            # 워커 스레드 시작
            processing_thread = threading.Thread(target=self._process_frame_worker, daemon=True)
            processing_thread.start()
            
            metrics_thread = threading.Thread(target=self._metrics_sender_worker, daemon=True)
            metrics_thread.start()
            
            self.socketio.run(self.app, host=host, port=port, debug=debug, use_reloader=False)
        except KeyboardInterrupt:
            print("\n🛑 서버 종료 중...")
            self.metrics.processing_active = False
            self.camera.release()
        except Exception as e:
            print(f"❌ 서버 시작 실패: {e}")
            self.metrics.processing_active = False
            self.camera.release()


class InterviewMetrics:
    """면접 분석 메트릭 클래스"""
    def __init__(self):
        # 조정 가능한 임계값들
        self.ear_threshold = 0.12
        self.ear_consecutive_frames = 1
        self.ear_calibration_frames = 30
        
        self.nod_threshold = 15
        self.shake_threshold = 25
        self.nod_cooldown = 1.0
        self.shake_cooldown = 1.0
        self.head_min_frames = 8
        self.nod_pattern_frames = 4
        
        self.movement_threshold = 0.06
        self.posture_min_frames = 10
        self.posture_sway_cooldown = 2.0
        
        self.smile_threshold = 5
        self.smile_intensity_high = 10
        
        self.hand_gesture_threshold = 0.03
        self.hand_gesture_cooldown = 1.5
        self.hand_min_frames = 6
        self.hand_analysis_interval = 5
        
        self.enable_hand_analysis = True
        self.frame_skip_base = 2
        self.frame_skip_with_hands = 3
        
        # 시스템 변수들
        self.analyzing = False
        self.analysis_start_time = None  # 분석 시작 시간
        self.analysis_end_time = None    # 분석 종료 시간
        
        self.frame_skip = 0
        self.skip_interval = self.frame_skip_base
        
        # 상태 변수들
        self.blink_count = 0
        self.last_blink_time = 0
        self.ear_below = False
        self.ear_closed_frames = 0
        self.ear_history_for_calibration = deque(maxlen=self.ear_calibration_frames)
        self.is_ear_calibrated = False
        self.dynamic_ear_threshold = self.ear_threshold
        
        self.head_pose_history = deque(maxlen=15)
        self.nod_count = 0
        self.shake_count = 0
        self.last_nod_time = 0
        self.last_shake_time = 0
        self.pitch_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=5)
        
        self.emotion_history = deque(maxlen=5)
        self.micro_expression_count = 0
        
        self.smile_start_time = None
        self.smile_duration = 0
        self.total_smile_time = 0
        self.smile_intensity_history = deque(maxlen=30)
        
        self.shoulder_positions = deque(maxlen=30)
        self.posture_sway_count = 0
        self.last_sway_time = 0
        
        self.hand_positions = deque(maxlen=15)
        self.hand_gesture_count = 0
        self.last_gesture_time = 0
        self.hands_visible_count = 0
        self.hand_frame_counter = 0
        
        self.frame_count = 0
        self.start_time = time.time()
        self.processing_active = True


class OneEuroFilter:
    """One Euro Filter for smoothing"""
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


if __name__ == '__main__':
    analyzer = InterviewAnalyzer()
    analyzer.run()