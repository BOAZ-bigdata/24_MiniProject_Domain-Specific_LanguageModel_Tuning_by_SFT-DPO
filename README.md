# 24_MiniProject_Domain-Specific_LanguageModel_Tuning_by_SFT-DPO
SFT_DPO를 활용한 도메인 특화 QA Model 성능 개선 프로젝트

### 과정
>> SFT 및 DPO를 통해 특허 도메인 내에서의 기술적 Question And Answering 능력을 높이도록 함
>>> 이때, EXAONE model을 Baseline Model로 삼고, Fine tune 전 후를 비교하여 성능을 파악함
>>> QA set는 AI hub내 데이터 셋을 이용하여 생성하였으며, 검증을 위한 QA set은 Human Resource로 생성

### 결과
>> 기존 모델에 비해 약간의 성능 향상을 보임(~5%)
>>> Model 자체가 본 데이터셋을 학습했을 가능성이 있음


### 시사점
1. 특허 데이터셋 자체도 QA를 만들기 쉽지 않았음(덜 정형화되어있음) 타 Medical QA 셋과 같이 정제되지 못했을 가능성
2. SFT의 중요성-> DPO는 약간의 변주

- Team members: [[고혜정](https://github.com/Kohyejung)] ,[[신진섭](https://github.com/ShinJinSeob?query=%EC%8B%A0)]
