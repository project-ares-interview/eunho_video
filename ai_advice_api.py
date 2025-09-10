# ai_advice_api.py - AI 조언 API 엔드포인트 추가 코드
# ai_interview_coach.py에 다음 코드를 추가하세요

from flask import request, jsonify
from ai_interview_coach import app, socketio, metrics, camera
from openai_advisor import InterviewAdvisor
import dotenv
import os

# AI 조언 시스템 초기화 (전역 변수)
advisor = None

dotenv.load_dotenv('.env.keys', override=True)

def init_ai_advisor():
    """AI 조언 시스템 초기화"""
    global advisor
    try:
        advisor = InterviewAdvisor(
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            deployment_name=os.getenv('AZURE_OPENAI_MODEL', 'gpt-4')
        )
        print("🤖 AI 조언 시스템 초기화 완료")
        return True
    except Exception as e:
        print(f"❌ AI 조언 시스템 초기화 실패: {e}")
        return False

# ai_interview_coach.py의 라우트 섹션에 다음 코드를 추가:

@app.route('/api/generate_ai_advice', methods=['POST'])
def generate_ai_advice():
    """AI 기반 면접 조언 생성"""
    try:
        global advisor
        
        if not advisor:
            # AI 조언 시스템이 초기화되지 않은 경우
            return {
                'status': 'error',
                'message': 'AI 조언 시스템이 초기화되지 않았습니다.',
                'fallback_advice': '기본 조언을 사용합니다. Azure OpenAI 설정을 확인해주세요.'
            }
        
        # 요청 데이터 가져오기
        analysis_data = request.get_json()
        if not analysis_data:
            return {
                'status': 'error', 
                'message': '분석 데이터가 제공되지 않았습니다.'
            }
        
        print(f"🤖 AI 조언 생성 시작...")
        
        # AI 조언 생성
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
            'fallback_advice': '서버 오류로 인해 조언을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.'
        }

@app.route('/api/ai_status')
def ai_status():
    """AI 조언 시스템 상태 확인"""
    global advisor
    
    if not advisor:
        return {
            'status': 'not_initialized',
            'message': 'AI 조언 시스템이 초기화되지 않았습니다.',
            'has_api_key': bool(os.getenv('AZURE_OPENAI_API_KEY')),
            'has_endpoint': bool(os.getenv('AZURE_OPENAI_ENDPOINT'))
        }
    
    return {
        'status': 'ready',
        'message': 'AI 조언 시스템이 준비되었습니다.',
        'has_api_key': bool(advisor.api_key),
        'has_endpoint': bool(advisor.endpoint),
        'deployment_name': advisor.deployment_name
    }

@app.route('/api/test_ai_advice')
def test_ai_advice():
    """AI 조언 시스템 테스트용 엔드포인트"""
    global advisor
    
    if not advisor:
        return {
            'status': 'error',
            'message': 'AI 조언 시스템이 초기화되지 않았습니다.'
        }
    
    # 테스트 데이터
    test_data = {
        'session_info': {
            'duration_seconds': 120.0,
            'total_frames': 1200,
            'analysis_timestamp': '2025-09-09T16:13:00'
        },
        'behavioral_metrics': {
            'eye_contact': {
                'blink_count': 40,
                'blink_rate_per_minute': 20.0,
                'average_ear': 0.280
            },
            'facial_expressions': {
                'total_smile_time': 30.0,
                'smile_percentage': 25.0,
                'average_smile_intensity': 10.0,
                'micro_expressions': 5
            },
            'head_movements': {
                'nod_count': 8,
                'shake_count': 1,
                'head_stability_score': 85.0
            },
            'posture': {
                'sway_count': 2,
                'stability_score': 88.0
            },
            'hand_gestures': {
                'gesture_count': 10,
                'hands_visible_seconds': 80.0,
                'gesture_frequency_per_minute': 5.0
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
    
    try:
        result = advisor.generate_advice(test_data)
        return {
            'status': 'success',
            'test_result': result,
            'message': 'AI 조언 시스템 테스트 완료'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'AI 조언 테스트 실패: {str(e)}'
        }

# 서버 시작 시 AI 시스템 초기화
if __name__ == '__main__':
    print("🤖 AI 면접 코치 서버 초기화 중...")
    
    # 환경 변수 확인
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if api_key and endpoint:
        print("✅ Azure OpenAI 인증 정보 확인됨")
        init_success = init_ai_advisor()
        if init_success:
            print("🎯 AI 조언 기능 활성화")
        else:
            print("⚠️ AI 조언 기능 비활성화 (기본 조언 사용)")
    else:
        print("⚠️ Azure OpenAI 인증 정보 없음 - 기본 조언만 사용")
        print("환경 변수 설정:")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_DEPLOYMENT_NAME='gpt-4'")
    
    print("📊 기능:")
    print("- 🎯 실시간 행동 분석")  
    print("- 🕒 정확한 세션 시간 추적")
    print("- 🤖 AI 기반 면접 조언" + (" (활성화)" if api_key and endpoint else " (비활성화)"))
    print("- 📈 과학적 근거 기반 피드백")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 중...")
        metrics.processing_active = False
        camera.release()
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        metrics.processing_active = False
        camera.release()