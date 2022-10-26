# Derive-Emerging-Technologies

## 연구 목적: derive emerging technologies using patent map and ML
자세한 내용 설명 -> **GTM.pdf**

### data
- 출처: https://www.uspto.gov/
- 특정 산업에 관련된 특허를 수집
  - text 정보 수집(title, abstract)
  - citation 정보 수집

### process
<img width="733" alt="image" src="https://user-images.githubusercontent.com/46666833/197974393-5e38b59a-8414-4578-a89e-9b424935062a.png">

#### 1. 데이터 수집 및 전처리
- 전처리 과정
  - 소문자화, 영어가 아닌 문자 삭제
  - 불용어 제거
  - Lemmatize → 명사(NN, NNP)만 추출
  - Term Frequency 기준 상위 5% 단어를 keyword로 지정 추출된 keyword 에서 추가 불용어 제거
  - TF-IDF 행렬을 통해 patent-keyword matrix 구축
  
#### 2. 공백 영역 도출
- 차원 축소
  
  <img width="603" alt="image" src="https://user-images.githubusercontent.com/46666833/197975096-3cf42325-bf6d-490b-8a9c-eebf7db9b084.png">
- GTM 공백 영역 벡터 도출
  <img width="626" alt="image" src="https://user-images.githubusercontent.com/46666833/197975203-3b02bdde-958f-4277-a765-719d1665ffe6.png">

#### 3. 공백 영역 유망성 평가
- 비교 집단 정의
  - 유망 특허 집단, 주변 특허 집단 정의
- 공백 영역 역맵핑 결과로 공백에 대한 문장 생성
- 키워드 기반 분석, 지표 기반 분석, 머신러닝 기반 분석 수행

### Conclusion
- GTM 특허 지도를 통해 공백 기술 영역을 발굴하고 해당 공백 기술 영역이 유망한지 그렇지 않은지에 대한 평가를 통해 추후 개발이 필요한 기술 영역을 제시함
- 공백 추출에 초점을 맞춘 이전 연구에서 더 나아가 공백의 유망성을 평가함
  - 본 연구에서는 GTM 역맵핑 결과를 가지고 키워드 기반 분석, 지표 기반 분석, 머신러닝 기반 분석을 통해 종합적인 평가를 수행함
- 잠재적 기술 기회를 발굴함에 있어 중요한 시사점을 제공할 수 있음 

한계)
특정 산업을 중심으로 진행되기 때문에 추후 방법론의 적용 범위를 넓히는 것이 필요함
특허 벡터는 희소 행렬(Sparse matrix) 이기 때문에 차원 축소 시 정보의 손실이 존재함
특허의 피인용수가 높은 기술이 유망하다고 가정하고 분석함
예측된 유망 공백 영역에 대한 검증이 부족함
