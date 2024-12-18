# 🌸 마음 쉼터 상담 챗봇 🌸

## 서비스 소개
**마음 쉼터 상담 챗봇**은 사용자의 마음을 편안하게 해주는 대화형 AI 챗봇입니다. 불안, 스트레스, 공허함 등 다양한 감정을 나누고 싶을 때 언제든지 대화를 통해 위로와 조언을 받을 수 있도록 설계되었습니다. **LangChain**과 **Hugging Face**의 감정 분석 모델을 사용하여 사용자의 감정을 파악하고, 상황에 맞는 대답을 제공합니다.

이 챗봇은 직관적인 UI와 대화형 인터페이스를 통해 마치 친구와 이야기하듯 편안한 대화를 나눌 수 있도록 돕습니다. 또한, 상담 후에는 사용자의 피드백을 통해 챗봇의 성능과 사용자 경험을 지속적으로 개선합니다.

## 🛠️ 기술 스택
- **Python**: 언어와 데이터 처리에 사용
- **Streamlit**: 웹 인터페이스 구성
- **LangChain**: 자연어 처리와 문맥을 이해하는 대화 모델
- **Hugging Face**: 감정 분석 모델 (`nlptown/bert-base-multilingual-uncased-sentiment`)을 통해 사용자의 감정 상태를 파악
- **OpenAI GPT-4 모델**: 대화 응답 생성
- **dotenv**: 환경 변수 관리

## 📝 기능 설명

### 1. 감정 기반 응답 생성
사용자가 입력한 문장의 감정을 분석하여 그에 맞는 어조로 답변을 제공합니다. 감정 분석 모델을 통해 감정 상태를 파악하고, 챗봇이 상황에 맞는 말투(예: 공감적, 위로적)로 응답하도록 합니다.

### 2. 직관적인 대화형 UI
Streamlit으로 구축된 UI는 사용자가 편리하게 이용할 수 있도록 설계되었습니다. 사용자와 챗봇의 메시지는 각각 말풍선 형태로 표시되며, 사용자의 메시지는 우측, 챗봇의 응답은 좌측에 나타나 대화 흐름을 쉽게 따라갈 수 있습니다.

### 3. 상담 종료 후 피드백
대화가 끝난 후, **상담 종료** 버튼을 누르면 사용자에게 상담 만족도를 묻는 피드백 창이 표시됩니다. 사용자는 만족도 평가 후 **제출** 버튼을 누를 수 있으며, 제출 후 "피드백을 주셔서 감사합니다!"라는 메시지가 잠시 표시된 뒤 자동으로 사라집니다. 이를 통해 챗봇 서비스의 질을 향상시키고자 합니다.