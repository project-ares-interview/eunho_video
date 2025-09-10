import threading
import time
from queue import Queue, Empty

from app.models.metrics import InterviewMetrics
from app.services.analysis_service import process_frame

# 세션 관리를 위한 딕셔너리
client_sessions = {}

def create_session(sid):
    """새로운 클라이언트 세션을 생성하고 초기화합니다."""
    if sid in client_sessions:
        print(f"[{sid}] 경고: 이미 존재하는 세션입니다.")
        return

    print(f"[{sid}] 새로운 세션 생성 중...")
    client_sessions[sid] = {
        "metrics": InterviewMetrics(),
        "frame_queue": Queue(maxsize=2),
        "metrics_queue": Queue(maxsize=5),
        "processing_active": True,
        "processing_thread": None,
        "metrics_thread": None,
    }
    print(f"[{sid}] 세션 생성 완료. 현재 활성 세션: {len(client_sessions)}")

def get_session(sid):
    """세션 ID로 세션 정보를 반환합니다."""
    return client_sessions.get(sid)

def delete_session(sid):
    """클라이언트 세션을 정리하고 삭제합니다."""
    session = client_sessions.pop(sid, None)
    if session:
        session["processing_active"] = False
        print(f"[{sid}] 세션 정리 완료. 현재 활성 세션: {len(client_sessions)}")
    else:
        print(f"[{sid}] 경고: 삭제할 세션을 찾을 수 없습니다.")


def start_worker_threads(sid, socketio_instance):
    """세션별 워커 스레드를 시작합니다."""
    session = get_session(sid)
    if not session:
        print(f"[{sid}] 스레드를 시작할 세션을 찾을 수 없습니다.")
        return

    processing_thread = threading.Thread(
        target=process_frame_worker, args=(sid,), daemon=True
    )
    metrics_thread = threading.Thread(
        target=metrics_sender_worker, args=(sid, socketio_instance), daemon=True
    )

    session["processing_thread"] = processing_thread
    session["metrics_thread"] = metrics_thread

    processing_thread.start()
    metrics_thread.start()
    print(f"[{sid}] 워커 스레드 시작 완료.")


# 각 세션의 백그라운드 작업을 처리하는 워커 함수들
def process_frame_worker(sid):
    """별도 스레드에서 프레임 처리"""
    print(f"[{sid}] 프레임 처리 워커 시작")
    session = get_session(sid)
    if not session:
        print(f"[{sid}] 세션을 찾을 수 없어 워커를 종료합니다.")
        return

    while session.get("processing_active", False):
        try:
            frame = session["frame_queue"].get(timeout=1)
            if frame is None:
                break

            current_metrics, _ = process_frame(frame, session["metrics"])

            if not session["metrics_queue"].full():
                session["metrics_queue"].put(current_metrics)

            session["frame_queue"].task_done()

        except Empty:
            continue
        except Exception as e:
            print(f"[{sid}] Frame processing worker error: {e}")
            time.sleep(0.1)
    print(f"[{sid}] 프레임 처리 워커 종료")


def metrics_sender_worker(sid, socketio_instance):
    """메트릭을 Socket.IO로 전송하는 별도 스레드"""
    print(f"[{sid}] 메트릭 전송 워커 시작")
    session = get_session(sid)
    if not session:
        print(f"[{sid}] 세션을 찾을 수 없어 워커를 종료합니다.")
        return

    while session.get("processing_active", False):
        try:
            current_metrics = session["metrics_queue"].get(timeout=1)
            if current_metrics:
                # socketio.emit()을 직접 호출하는 대신, 주입된 인스턴스를 사용
                socketio_instance.emit("metrics_update", current_metrics, room=sid)
                session["metrics_queue"].task_done()
        except Empty:
            continue
        except Exception as e:
            print(f"[{sid}] Metrics sender error: {e}")
            time.sleep(0.1)
    print(f"[{sid}] 메트릭 전송 워커 종료")
