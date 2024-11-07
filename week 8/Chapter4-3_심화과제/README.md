# 🤖 나만의 LLM 서비스 경량화 프로젝트

## 개요
이번 프로젝트는 LLM(대형 언어 모델) 서비스의 학습 속도를 높이고 메모리 효율성을 개선하기 위해 **경량화 기법**을 적용한 실험입니다.  
**7주차에 개발한 모델은 주로 프롬프트 기반이어서 경량화 효과를 제대로 체감하기 어려울 것으로 판단하여, 대신 6주차에 사용된 학습 모델을 사용하여 진행하였습니다.** 학습 데이터는 `corpus.json`에 저장되어 있습니다.

## 📂 주요 파일 설명

총 4개의 Python 파일로 실험을 구성했습니다.

- `before_small.py` : 경량화 적용 전, `facebook/opt-350m` 모델을 사용한 코드
- `lightweight_small.py` : 경량화 적용 후, `facebook/opt-350m` 모델을 사용한 코드
- `before_big.py` : 경량화 적용 전, `EleutherAI/gpt-neo-1.3B` 모델을 사용한 코드
- `lightweight_big.py` : 경량화 적용 후, `EleutherAI/gpt-neo-1.3B` 모델을 사용한 코드

`before` 파일들은 경량화 적용 전 상태를, `lightweight` 파일들은 경량화를 적용한 상태를 의미합니다. `small` 파일들은 작은 모델(`350M`)을, `big` 파일들은 큰 모델(`1.3B`)을 사용하여 실험했습니다.

## 🧪 실험 과정 및 결과

### 1. 작은 모델 실험 (`facebook/opt-350m`)

작은 모델(`350M`)을 사용해 경량화의 효과를 먼저 실험했습니다. `before_small.py`와 `lightweight_small.py`를 각각 실행하여 학습 시간을 비교해보았습니다.

- **경량화 전 학습 시간** : 2.67분  
  ![image](https://github.com/user-attachments/assets/8965bcb3-a7a7-44e0-aba5-139233f55fa2)

- **경량화 후 학습 시간** : 0.72분 (약 3배 속도 향상 🚀)  
  ![image](https://github.com/user-attachments/assets/ec223c43-d8b2-42dc-8c36-ef62d5802be1)

경량화를 통해 **학습 속도가 약 3배 빨라졌으며**, `train/loss` 그래프를 통해 학습 손실이 안정적으로 감소하는 것을 확인했습니다. 성능 저하 없이 속도를 크게 개선한 결과입니다.  

또한 경량화 전후 모두 안정적으로 `loss`가 줄어드는 모습이 관찰되었습니다.

- **경량화 전 Loss 감소 추이**:  
  ![image](https://github.com/user-attachments/assets/e1373019-e4c0-4ab7-abea-efab81687385)

- **경량화 후 Loss 감소 추이**:  
  ![image](https://github.com/user-attachments/assets/554b59eb-684c-41b5-9d70-219d9321151d)

### 2. 큰 모델 실험 (`EleutherAI/gpt-neo-1.3B`)

더 큰 모델인 `1.3B` 파라미터 모델로 실험을 진행했습니다. 경량화 적용 전 파일(`before_big.py`)을 실행하려 했으나, **CUDA 메모리 부족 에러**가 발생해 학습이 불가능했습니다.  
![image](https://github.com/user-attachments/assets/eafb76a7-8fd6-4ada-9eb7-4a3e1dd757c9)

이를 해결하기 위해 `lightweight_big.py`에서는 경량화 기법을 적용하여 GPU 메모리를 효율적으로 사용하도록 변경했습니다.

#### 경량화 적용 방식
- **4비트 양자화** : 모델 파라미터를 4비트로 줄여 메모리 사용량 감소
- **LoRA (Low-Rank Adaptation)** : 모델의 일부 모듈에만 학습 가능한 파라미터를 추가하여 경량화

경량화를 통해 학습을 성공적으로 실행할 수 있었으며, 이를 통해 메모리 효율성의 이점을 확인했습니다.  
![image](https://github.com/user-attachments/assets/1375ea42-d610-41ac-ba0d-1e462459bd49)


## 💡 얻은 이점
경량화 기법을 통해 다음과 같은 이점을 얻을 수 있었습니다.

1. **학습 속도 개선** : 작은 모델(`350M`)에서 경량화 후 약 3배 빠른 학습 속도를 확인했습니다.
2. **메모리 효율성 향상** : 큰 모델(`1.3B`)은 경량화 없이는 실행이 불가능했으나, 양자화와 LoRA 기법을 통해 메모리 제한 문제를 해결했습니다.

이와 같은 결과를 통해 **경량화 기법**이 대형 모델 학습에 필수적이라는 점을 실감할 수 있었습니다. 특히, 제한된 하드웨어 환경에서 더 큰 모델을 사용할 수 있는 가능성을 보여줬습니다.
