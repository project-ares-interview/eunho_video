from app.main import app, socketio

if __name__ == "__main__":
    print("🚀 Flask 서버 시작...")
    try:
        socketio.run(app, host="0.0.0.0", port=5001, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
