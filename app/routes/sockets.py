import base64
from datetime import datetime
import time

import numpy as np
import cv2
from flask import request
from flask_socketio import emit

from app.models.metrics import InterviewMetrics
from app.services.session_manager import (
    create_session,
    delete_session,
    get_session,
    start_worker_threads,
)
from app.services.analysis_service import get_detailed_analysis_data


def register_socket_handlers(socketio, advisor):
    """SocketIO 이벤트 핸들러를 등록하는 함수"""

    @socketio.on_error()
    def error_handler(e):
        print(f"❌ SocketIO Error: {e}")
        return True

    @socketio.on("connect")
    def handle_connect():
        sid = request.sid
        print(f"🌐 클라이언트 연결됨: {sid}")
        create_session(sid)
        start_worker_threads(sid, socketio)
        emit("connection_status", {"status": "connected", "timestamp": time.time()})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = request.sid
        print(f"🌐 클라이언트 연결 끊김: {sid}")
        delete_session(sid)

    @socketio.on("video_frame")
    def handle_video_frame(data):
        sid = request.sid
        session = get_session(sid)
        if not session:
            return

        try:
            image_data = data.get("image")
            if image_data:
                header, encoded = image_data.split(",", 1)
                decoded_image = base64.b64decode(encoded)
                nparr = np.frombuffer(decoded_image, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_queue = session["frame_queue"]
                    if frame_queue.full():
                        frame_queue.get_nowait()
                    frame_queue.put(frame)
                else:
                    print(f"[{sid}] ⚠️ 빈 프레임을 수신했습니다.")
        except Exception as e:
            print(f"[{sid}] ❌ 비디오 프레임 처리 오류: {e}")

    @socketio.on("toggle_analysis")
    def toggle_analysis(data):
        sid = request.sid
        session = get_session(sid)
        if not session:
            return
        try:
            metrics = session["metrics"]
            analyzing = data.get("analyze", False)
            session_duration = 0
            if analyzing:
                metrics.analysis_start_time = time.time()
                metrics.analysis_end_time = None
            else:
                metrics.analysis_end_time = time.time()
                if metrics.analysis_start_time:
                    session_duration = (
                        metrics.analysis_end_time - metrics.analysis_start_time
                    )
            metrics.analyzing = analyzing
            emit(
                "analysis_status",
                {
                    "analyzing": metrics.analyzing,
                    "timestamp": time.time(),
                    "session_duration": session_duration
                    if not analyzing and metrics.analysis_end_time
                    else 0,
                },
            )
        except Exception as e:
            print(f"[{sid}] Toggle analysis error: {e}")
            emit("error", {"message": f"Analysis toggle failed: {str(e)}"})

    @socketio.on("reset_metrics")
    def reset_metrics_socket():
        sid = request.sid
        session = get_session(sid)
        if not session:
            return
        try:
            metrics = session["metrics"]
            old_analyzing = metrics.analyzing
            old_hand_enabled = metrics.enable_hand_analysis
            session["metrics"] = InterviewMetrics()
            session["metrics"].analyzing = old_analyzing
            session["metrics"].enable_hand_analysis = old_hand_enabled
            emit(
                "reset_metrics_response",
                {"status": "success", "message": "Metrics reset successfully"},
            )
        except Exception as e:
            print(f"[{sid}] Reset metrics error: {e}")
            emit("reset_metrics_response", {"status": "error", "message": str(e)})

    @socketio.on("get_summary")
    def get_summary_socket():
        sid = request.sid
        session = get_session(sid)
        if not session:
            emit("get_summary_response", {"status": "error", "message": "Session not found"})
            return
        try:
            metrics = session["metrics"]
            elapsed_time = time.time() - metrics.start_time
            analysis_session_time = 0
            if metrics.analysis_start_time:
                if metrics.analysis_end_time:
                    analysis_session_time = (
                        metrics.analysis_end_time - metrics.analysis_start_time
                    )
                else:
                    analysis_session_time = time.time() - metrics.analysis_start_time
            hands_visible_seconds = (
                round((metrics.hands_visible_count / 20), 1)
                if metrics.frame_count > 0
                else 0
            )
            summary_data = {
                "session_duration": round(elapsed_time, 1),
                "analysis_session_duration": round(analysis_session_time, 1),
                "total_frames": metrics.frame_count,
                "blink_rate": round(metrics.blink_count / (analysis_session_time / 60), 1)
                if analysis_session_time > 0
                else 0,
                "nod_count": metrics.nod_count,
                "shake_count": metrics.shake_count,
                "micro_expressions": metrics.micro_expression_count,
                "total_smile_time": round(metrics.total_smile_time, 1),
                "posture_sway_count": metrics.posture_sway_count,
                "hand_gesture_count": metrics.hand_gesture_count,
                "hands_visible_seconds": hands_visible_seconds,
                "hands_visible_rate": round(
                    (metrics.hands_visible_count / metrics.frame_count) * 100, 1
                )
                if metrics.frame_count > 0
                else 0,
                "avg_smile_intensity": round(
                    np.mean(list(metrics.smile_intensity_history))
                    if metrics.smile_intensity_history
                    else 0,
                    2,
                ),
            }
            emit("get_summary_response", {"status": "success", "data": summary_data})
        except Exception as e:
            print(f"[{sid}] Get summary error: {e}")
            emit("get_summary_response", {"status": "error", "message": str(e)})
    
    @socketio.on("generate_ai_advice")
    def generate_ai_advice_socket():
        sid = request.sid
        session = get_session(sid)
        if not session:
            emit("generate_ai_advice_response", {"status": "error", "message": "Session not found"})
            return
        try:
            if not advisor:
                emit(
                    "generate_ai_advice_response",
                    {
                        "status": "error",
                        "message": "AI 조언 시스템이 초기화되지 않았습니다.",
                        "fallback_advice": "기본 조언을 사용합니다. Azure OpenAI 설정을 확인해주세요.",
                    },
                )
                return
            analysis_data = get_detailed_analysis_data(session["metrics"])
            if not analysis_data:
                emit(
                    "generate_ai_advice_response",
                    {"status": "error", "message": "분석 데이터가 제공되지 않았습니다."},
                )
                return
            advice_result = advisor.generate_advice(analysis_data)
            if advice_result["status"] == "success":
                emit(
                    "generate_ai_advice_response",
                    {
                        "status": "success",
                        "advice": advice_result["advice"],
                        "analysis_summary": advice_result.get("analysis_summary", {}),
                        "timestamp": advice_result["timestamp"],
                    },
                )
            else:
                emit(
                    "generate_ai_advice_response",
                    {
                        "status": "error",
                        "message": advice_result.get("message", "AI 조언 생성 실패"),
                        "fallback_advice": advice_result.get(
                            "fallback_advice", "기본 조언을 생성할 수 없습니다."
                        ),
                    },
                )
        except Exception as e:
            print(f"[{sid}] ❌ AI 조언 API 오류: {e}")
            emit(
                "generate_ai_advice_response",
                {
                    "status": "error",
                    "message": str(e),
                    "fallback_advice": "서버 오류로 인해 조언을 생성할 수 없습니다.",
                },
            )
