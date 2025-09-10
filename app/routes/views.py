from flask import Blueprint, render_template, jsonify
from app.services.session_manager import client_sessions
import time

# Blueprint를 생성하여 라우트를 체계적으로 관리합니다.
views = Blueprint('views', __name__)

@views.route("/")
def index():
    """메인 인터뷰 코치 페이지를 렌더링합니다."""
    return render_template("interview_coach.html")

@views.route("/api/health")
def health_check():
    """서버의 상태와 활성 세션 수를 반환합니다."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "active_sessions": len(client_sessions),
    })
