import os
import dotenv
from flask import Flask
from flask_socketio import SocketIO

from openai_advisor import InterviewAdvisor
from app.routes.views import views
from app.routes.sockets import register_socket_handlers

# .env 파일에서 환경 변수 로드
dotenv.load_dotenv(".env.keys", override=True)

# AI 조언자 인스턴스
advisor = None

def init_ai_advisor():
    """AI 조언 시스템을 초기화합니다."""
    global advisor
    try:
        api_key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")

        if not all([api_key, endpoint, deployment_name]):
            print("⚠️ AI 조언 시스템 환경 변수가 설정되지 않았습니다. 기능이 비활성화됩니다.")
            return False

        advisor = InterviewAdvisor(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
        )
        print("🤖 AI 조언 시스템 초기화 완료")
        return True
    except Exception as e:
        print(f"❌ AI 조언 시스템 초기화 실패: {e}")
        return False

# Flask 앱과 SocketIO 인스턴스 생성
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=30,
    ping_interval=10,
    logger=False,
    engineio_logger=False,
)

# AI 어드바이저 초기화 실행
init_ai_advisor()

# HTTP 라우트(Blueprint) 등록
app.register_blueprint(views)

# SocketIO 이벤트 핸들러 등록
register_socket_handlers(socketio, advisor)
