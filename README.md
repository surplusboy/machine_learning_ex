# 머신러닝의 하위장르 딥러닝을 배워보는 장소  

뉴럴 네트워크 (신경망)을 이용해 머신러닝을 진행하는게 딥러닝   
전통적인 머신러닝 방법론에 비하여 해결하고자 하는 문제에 대한 사전 지식 (Domain Knowledge)이 덜 요구되는것이 장점이다.

딥러닝을 적용할 수 있는 분야
1. image classification / object detection
2. sequence data 분석 & 예측


GPU로 딥러닝을 돌리기 위해 필요한 어플리케이션
1. NVIDIA 그래픽 드라이버
2. CUDA
3. cuDNN

머신러닝의 종류  
1. Supervised Learning : 데이터에 정답이 있고 정답 예측 모델을 만들 때 ex) 이미지 분류 (강아지, 고양이사진을 분류)
2. Unsupervised Learning : 데이터에 정답이 없는 예측 모델을 만들 때 ex) 추천 알고리즘 (옷, 영화 추천 등 -> 군집합)
3. Reinforcement Learning : 강화학습 어떠한 환경에서 어떠한 행동의 결과로 보상과 벌칙을 부여해 반복을 통해 스스로 학습 (trial and error)

간단한 머신러닝 예시

```commandline
수능점수 예측 모델
전제 : 모의 고사 6월성적과 9월 성적은 수능성적에 절반씩 영향이 있을것 같다
수식 : 6월 성적 * w1 + 9월 성적 * w2 + b = 수능 성적
```
w : weight (가충치)  
b : bias (편향), w1, w2에는 관계 없지만 결과값에는 관계성이 있는 요소(상수)

퍼셉트론과 신경망  

신경망을 통해 학습 결과 -> feature extraction (특성 추출)
평균 제곱 오차 : 각 데이터 결과값의 오차를 제곱하여 평균을 낸 값 -> Loss Function, 정수 예측시 주로 쓰임  
Loss Function : 모델의 정확도를 측정하기 위한 함수 계산식

local minima

경사 하강법에 위해 최소 오차값을 찾을때 지역 최소값을 넘어 전역 최소값을 찾기 위해 learning rate를 지정한다.

learning rate optimizer  

좋은 학습을 위헤 learning rate 를 고정적이지 않고 가변적으로 변경할 수 있는 알고리즘이 여러개 있다.
- momentum : 가속도를 유지하며 경사 하강
- AdaGrad : 자주변하는 w는 작게, 자주 안변하면 크게 (local minima에 빠지는걸 방지)
- RMSProp : AdaGrad를 제곱
- AdaDelta : AdaGrad의 a(learning rate를) 값이 너무 작아져서 학습되지 않는걸 방지
- Adam : RMSProp + Momentum -> 일반적으로 사용하게 되는 옵티마이저