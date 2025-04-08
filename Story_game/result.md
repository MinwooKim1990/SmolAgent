# RAG 시스템 비교 분석 보고서

## 1. 알고리즘 파이프라인 설명

### 1.1 GraphRAG (ImprovedGraphRAG)
GraphRAG는 스토리를 지식 그래프로 변환하여 처리하는 방식을 사용합니다.

**주요 파이프라인:**
1. **스토리 입력 및 전처리**
   - 스토리 텍스트 입력
   - 해시 기반 캐시 확인

2. **지식 그래프 생성**
   - GraphBuilder 에이전트가 스토리를 분석
   - 노드(등장인물, 장소, 사건 등) 추출
   - 관계(엣지) 정의
   - 각 노드와 관계에 대한 상세 설명 포함

3. **그래프 정제**
   - GraphRefiner 에이전트가 그래프 구조 개선
   - 유사 노드 병합
   - 누락된 관계 추가
   - 중복/불필요한 연결 제거

4. **임베딩 및 인덱싱**
   - 노드와 관계 정보를 벡터로 변환
   - FAISS를 사용한 벡터 인덱싱

5. **질의응답 처리**
   - 하이브리드 검색 (벡터 유사도 + 그래프 구조)
   - 관련 서브그래프 추출
   - QA 에이전트를 통한 답변 생성

### 1.2 StandardRAG
StandardRAG는 전통적인 청크 기반 RAG 시스템입니다.

**주요 파이프라인:**
1. **스토리 입력 및 전처리**
   - 스토리 텍스트 입력
   - 해시 기반 캐시 확인

2. **청크 분할**
   - TextChunker 에이전트가 스토리를 의미 있는 단위로 분할
   - 문맥을 유지하면서 적절한 크기로 분할

3. **임베딩 및 인덱싱**
   - 각 청크를 벡터로 변환
   - FAISS를 사용한 벡터 인덱싱

4. **질의응답 처리**
   - 벡터 유사도 기반 관련 청크 검색
   - Self-Consistency 적용 (여러 답변 생성 후 최적 답변 선택)
   - QA 에이전트를 통한 답변 생성

## 2. 실험 및 평가

### 2.1 테스트 데이터셋

#### 영어 텍스트
```text
"""Alice is a curious girl who discovers Wonderland. 
    She follows a White Rabbit down a rabbit hole and encounters various characters. 
    The Mad Hatter hosts a tea party with the March Hare. 
    The Queen of Hearts rules Wonderland with strict authority."""
```

#### 한국어 텍스트
```text
"""미래 도시 서울의 중심가에 위치한 '디지털 숲' 공원은 최첨단 생태 기술의 집약체입니다. 
    이곳의 나무들은 대기 오염을 실시간으로 측정하고 정화하는 나노 센서를 갖추고 있으며, 시민들은 증강현실 앱을 통해 각 나무의 환경 기여도를 확인할 수 있습니다. 
    공원 곳곳에 설치된 '스마트 벤치'는 태양광으로 작동하며, 시민들에게 무선 충전과 환경 정보를 제공합니다.
    최근에는 인공지능 생태계 관리 시스템 '그린마인드'가 도입되어, 식물들의 생장 상태와 수분 공급을 자동으로 조절합니다. 
    시민들은 모바일 투표를 통해 공원의 새로운 식물 선정에 참여할 수 있으며, 
    계절별로 진행되는 환경 교육 프로그램은 증강현실 기술을 활용해 미래 환경의 중요성을 체험적으로 전달합니다."""
```

### 2.2 Standard RAG 평가 결과

#### 영어 텍스트 질의응답
Q: What role does the White Rabbit play in the story?
A: The White Rabbit plays the role of a guide that leads Alice into Wonderland by following him down a rabbit hole. He serves as the catalyst for Alice's adventure and her discovery of the various characters and experiences in Wonderland.

Q: How does Alice interact with the Mad Hatter?
A: <code style="background-color: lightbrown;">Alice interacts with the Mad Hatter</code> during a tea party that he hosts with the March Hare.

Q: What is the power structure in Wonderland?
A: The power structure in Wonderland is depicted as one where the Queen of Hearts rules with strict authority, suggesting a hierarchical system where she holds the ultimate power over the characters in Wonderland.

#### 한국어 텍스트 질의응답
Q: 스마트 벤치의 주요 기능은 무엇인가요?
A: 스마트 벤치의 주요 기능은 <code style="background-color: lightbrown;">실시간 정보 제공</code>과 환경 감지 기능을 갖추고 있으며, 주변의 다양한 데이터를 수집하여 분석하는 것이 포함됩니다. <code style="background-color: lightbrown;">이 벤치는 대기 오염, 환경 변화 등을 모니터링하며, 이를 통해 지역 사회의 다양한 환경 정보를 체계적으로 정리하고 제공하는 역할을 합니다.</code>

Q: 디지털 숲 공원의 연간 방문객 수는 얼마나 되나요?
A: 제공된 맥락에서 디지털 숲 공원의 연간 방문객 수에 대한 정보는 포함되어 있지 않습니다. 따라서 해당 질문에 대한 답변을 드릴 수 없습니다.

Q: 디지털 숲 공원은 환경 교육에 어떤 영향을 미칠 것으로 예상되나요?
A: 디지털 숲 공원은 환경 교육에 긍정적인 영향을 미칠 것으로 예상됩니다. 특히, 막대한 데이터를 활용하여 환경 문제와 관련된 교육을 보다 효과적으로 진행할 수 있으며, 이러한 경험은 학생들에게 환경 변화에 대한 인식을 높이고 지속 가능한 발전의 필요성을 이해시키는데 도움을 줄 수 있습니다.

### 2.3 GraphRAG 평가 결과

#### 영어 텍스트 질의응답
Q: What role does the White Rabbit play in the story?
A: The White Rabbit plays the role of a guide in the story, as he leads Alice into Wonderland. He serves as the character that sparks Alice's curiosity and initiates her adventure in the fantastical location.

Q: How does Alice interact with the Mad Hatter?
A: The context does not provide a direct interaction between Alice and the Mad Hatter. However, both characters are part of the Wonderland universe, where the Mad Hatter hosts a tea party with his companion, the March Hare. Alice's main involvement is discovering Wonderland and following the White Rabbit, which indirectly suggests she may encounter the Mad Hatter at the tea party, <code style="background-color: blue;">but specific interactions between Alice and the Mad Hatter are not detailed in the provided information.</code>

Q: What is the power structure in Wonderland?
A: The power structure in Wonderland is primarily dominated by the Queen of Hearts, who rules over Wonderland with strict authority. She is central to the governance and control of the location. Alice, as a curious adventurer, discovers this fantastical place and interacts with various characters, including the Queen, but she does not hold power. The White Rabbit leads Alice into Wonderland, indicating a role of guidance rather than authority. The Mad Hatter and March Hare are part of a separate social dynamic, as they host tea parties but do not influence the power dynamics of Wonderland significantly. Overall, the Queen of Hearts is the key figure in the power hierarchy within Wonderland.

#### 한국어 텍스트 질의응답
Q: 스마트 벤치의 주요 기능은 무엇인가요?
A: 스마트 벤치의 주요 기능은 다음과 같습니다:

1. **정보 제공**: <code style="background-color: lightbrown;">스마트 벤치는 주변 환경 및 장소에 대한 정보를 제공하는 역할을 합니다.</code>
2. **충전 기능**: 스마트 벤치에는 USB 포트 등의 충전 기능이 있어 사용자가 전자 기기를 충전할 수 있도록 지원합니다.
3. **센서 기능**: <code style="background-color: lightbrown;">내부 센서를 통해 날씨 정보를 수집하고 주변 상황을 모니터링합니다.</code>

이러한 기능들은 사용자에게 더 나은 편의성과 연결성을 제공합니다.

Q: 디지털 숲 공원의 연간 방문객 수는 얼마나 되나요?
A: 주어진 컨텍스트에서 "디지털 숲 공원"에 대한 구체적인 방문객 수에 대한 정보는 제공되지 않았습니다. 따라서, 연간 방문객 수를 확인할 수 없습니다. 추가적인 데이터가 필요합니다.

Q: 디지털 숲 공원은 환경 교육에 어떤 영향을 미칠 것으로 예상되나요?
A: 디지털 숲 공원은 환경 교육에 긍정적인 영향을 미칠 것으로 예상됩니다. 해당 공원이 자연과 기술을 연결하고, 사용자가 환경 문제를 이해하고 해결책을 배울 수 있는 공간으로 기능할 수 있기 때문입니다. 공원 내의 다양한 구성 요소는 정보 제공, 체험 학습, 그리고 참여적 활동을 통해 방문객들에게 환경 교육의 기회를 제공합니다. 이러한 교육적 요소는 사람들이 지속 가능성과 생태계 보전의 중요성을 배우는 데 도움을 줄 것입니다.

### 2.4 평가 분석

#### 영어 텍스트 평가
1. **사실 기반 질문**: StandardRAG는 자신이 아는 지식에서 대답하는 경향이 있으나, GraphRAG는 주어진 정보만을 기반으로 답변
2. **추론 필요 질문**: GraphRAG가 더 신중하고 명확한 답변 제공
3. **불확실한 정보**: GraphRAG가 더 명확하게 불확실성을 표현하고 구분

#### 한국어 텍스트 평가
1. **사실 기반 질문**: 두 시스템 모두 불확실한 부분에 대해 적절히 표현
2. **누락 정보 질문**: 두 시스템 모두 적절히 처리
3. **추론 질문**: 두 시스템 모두 합리적인 추론 제공

### 2.5 성능 메트릭스 평가

#### 자동 평가 결과 (15문제 기준)
![Image](https://github.com/user-attachments/assets/504e81f8-0d45-4d05-b5c1-33c3f5be1996)
![Image](https://github.com/user-attachments/assets/8e47fb52-ae40-4bfd-8c96-0f1a5c5214ee)
![Image](https://github.com/user-attachments/assets/b4c6739b-5d8f-41c2-aca9-c68f3d4f9855)

- F1 스코어: StandardRAG가 약간 우세
- BLEU 점수: StandardRAG가 약간 우세
- ROUGE 점수: 비슷한 수준

### 2.6 LLM 평가

#### Claude 3.5 Sonnet 평가
![Image](https://github.com/user-attachments/assets/4a4f8255-4651-40c7-b0c8-247777fd3717)
*문제, 질문 및 정답 자료의 경우 claude 3.5 sonnet에 의해 제작되고 평가됨.

#### o3-Mini-high 평가 결과
| 번호 | 질문 항목 | ImprovedGraphRAG 점수 | StandardRAG 점수 |
|------|-----------|----------------------|------------------|
| 1 | 기계학습의 주요 유형 | 10 | 10 |
| 2 | 지도학습 vs 비지도학습 차이와 적합 상황 | 10 | 9 |
| 3 | 기계학습 방법의 한계와 발전 방향 | 9 | 8 |
| 4 | 파리협정 온도 제한 목표 | 6 | 9 |
| 5 | 극지방 기후변화 영향 및 전 지구적 결과 | 9 | 8 |
| 6 | 기후변화 대응 정책 및 기술 평가, 개선 방안 | 9 | 8 |
| 7 | 양자컴퓨팅의 기본 양자 특성 | 10 | 10 |
| 8 | 양자컴퓨팅의 암호체계 영향 | 9 | 9 |
| 9 | 양자컴퓨팅 타임라인 및 임시 솔루션 | 8 | 8 |
| 10 | 인간 뇌의 뉴런 수와 통신 방식 | 10 | 10 |
| 11 | 신기술 학습이 뇌 구조에 미치는 변화 | 10 | 10 |
| 12 | 뇌 네트워크와 의식 이론이 AI 개발에 미치는 영향 | 9 | 8 |
| 13 | 자연 선택에 필요한 4가지 요소 | 10 | 10 |
| 14 | 진화 증거들 간의 상호 보완성 | 10 | 10 |
| 15 | 진화 메커니즘을 활용한 글로벌 도전 과제 대응 | 9 | 9 |
| **평균** |  | **9.2** | **9.1** |

#### DeepSeek R1 평가 결과
| **질문 번호** | **질문 주제** | **GraphRAG 점수** | **StandardRAG 점수** | **비고** |
|--------------|--------------|------------------|-------------------|----------|
| 1 | 기계학습 주요 유형 | 4.5 | 4.0 | GraphRAG: 용어 차이 있으나 구조화 우수 / StandardRAG: 불필요 주석 포함 |
| 2 | 지도 vs. 비지도학습 차이 | 4.8 | 4.2 | GraphRAG: 세부 예시 및 키워드 강조 / StandardRAG: 심층 분석 부족 |
| 3 | 기계학습 한계점 및 발전 방향 | 4.7 | 4.0 | GraphRAG: 구체적 해결책 제시 / StandardRAG: 추상적 제안 |
| 4 | 파리협정 기온 상승 목표 | 3.0 | 4.5 | GraphRAG: 1.5°C만 강조 / StandardRAG: 1.5°C 및 2°C 모두 명시 |
| 5 | 극지방 생태계 영향 | 4.5 | 3.8 | GraphRAG: 구체적 영향 설명 / StandardRAG: 일반적 설명 |
| 6 | 기후변화 대응 정책 평가 | 4.6 | 4.0 | GraphRAG: 실행 방안 제시 / StandardRAG: 추상적 접근 |
| 7 | 양자컴퓨팅의 암호화 영향 | 4.9 | 4.5 | GraphRAG: 기술적 용어 설명 / StandardRAG: 핵심 포함하나 심도 부족 |
| 8 | 양자컴퓨팅 실용화 타임라인 | 4.7 | 4.3 | GraphRAG: 구체적 타임라인 및 전략 / StandardRAG: 타임라인 불명확 |
| 9 | 뉴런 수 및 의사소통 방식 | 5.0 | 5.0 | 동일한 정확성 |
| 10 | 신기술 학습의 뇌 구조 영향 | 4.8 | 4.5 | GraphRAG: 전문 용어 활용 / StandardRAG: 기본 메커니즘 설명 |
| 11 | 뇌 네트워크 조직이 AI에 미치는 영향 | 4.9 | 4.3 | GraphRAG: 구체적 이론 적용 / StandardRAG: 일반적 응용 분야 언급 |
| 12 | 자연선택의 4가지 요소 | 4.8 | 4.5 | GraphRAG: "선택 압력" 강조 / StandardRAG: 간결하나 일부 설명 생략 |
| 13 | 진화론 증거 상호보완성 | 4.7 | 4.2 | GraphRAG: 통합적 설명 / StandardRAG: 독립적 역할 설명에 그침 |
| 14 | 진화 메커니즘의 현실 문제 적용 | 4.8 | 4.3 | GraphRAG: 구체적 전략 제시 / StandardRAG: 실행 가능성 설명 미흡 |

---

### **종합 평가**

| **평가 항목** | **GraphRAG** | **StandardRAG** |
| --- | --- | --- |
| **평균 점수** | **4.63** | **4.26** |
| **장점** | - 심층 분석- 구체적 예시 및 전략- 전문 용어 활용 | - 간결성- 신뢰성 (문맥 한계 명시) |
| **단점** | - 일부 핵심 내용 생략 (예: 파리협정 2°C) | - 추상적 제안- 예시 및 실행 계획 부족 |

LLM의 contextual evaluation에 있어서는 GraphRAG이 조금 더 앞서는 것을 볼 수 있다.

## 3. 결론 및 시사점

### 3.1 주요 발견사항
1. **처리 시간**: GraphRAG가 StandardRAG보다 약 3배 더 소요
2. **답변 품질**: GraphRAG가 더 상세하고 구조화된 답변 제공
3. **한국어 처리**: 영어 기반 임베딩 모델 사용으로 한국어 처리에 제한

### 3.2 시스템별 특징

#### GraphRAG
- **장점**:
  - 구조화된 지식 표현
  - 신중한 추론
  - 명확한 불확실성 표현
- **단점**:
  - 긴 처리 시간
  - 일부 단순 질문에서 과도한 복잡성

#### StandardRAG
- **장점**:
  - 빠른 처리 속도
  - 단순 질문에 효율적
  - 구현 용이성
- **단점**:
  - 복잡한 관계 파악 제한
  - 맥락 이해 제한적