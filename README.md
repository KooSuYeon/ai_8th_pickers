# ai_8th_pickers
AI_8기 3주차 3조 pickers 리포입니다.

- [X] titanic 필수 과제 구현  
- [X] netflix 평점 예측 도전 과제 구현  
- [X] netflix sentiment 추가 도전 과제 구현  

---
- 깃 레포 연결 후에는 git pull origin main을 해주어야 합니다.
- 개인 브랜치 -> main PR(Pull Request) 생성 후 merge 원칙

<details>
    <summary>
    GITHUB COMMIT RULE
    </summary>

    - feat 		: 새로운 기능 추가
    - fix 		: 버그 수정
    - docs 		: 문서 수정
    - style 	: 코드 formatting, 세미콜론(;) 누락, 코드 변경이 없는 경우
    - refactor 	: 코드 리팩토링
    - test 		: 테스트 코드, 리팽토링 테스트 코드 추가
    - chore 	: 빌드 업무 수정, 패키지 매니저 수정
</details>

---
**필수과제 역할 분담**

#### 1. 데이터 셋 불러오기 및 feature 분석
- 담당 : 고준원

<details>
    <summary>
    Process
    </summary>

    1. seaborn 라이브러리에 있는 titanic 데이터를 불러옵니다.
    2-1. 데이터의 feature를 파악하기 위해 아래의 다양한 feature 분석을 수행해주세요. 
    2-2. describe 함수를 통해서 기본적인 통계를 확인해주세요. 
    2-3. describe 함수를 통해 확인할 수 있는 count, std, min, 25%, 50%, 70%, max 가 각각 무슨 뜻인지 주석 혹은 markdown 블록으로 간단히 설명해주세요. 
    2-4. isnull() 함수와 sum()  함수를 이용해 각 열의 결측치 갯수를 확인해주세요. 
</details>

#### 2. Feture Engineering
- 담당 : 박수호

<details>
    <summary>
    Process
    </summary>
    
    1-1. 결측치 처리 : Age(나이)의 결측치는 중앙값으로, Embarked(승선 항구)의 결측치는 최빈값으로 대체해주세요. 모두 대체한 후에, 대체 결과를 isnull() 함수와 sum()  함수를 이용해서 확인해주세요. 
    1-2. Sex(성별)를 남자는 0, 여자는 1로 변환해주세요. alive(생존여부)를 True는 1, False는 0으로 변환해주세요. Embarked(승선 항구)는 ‘C’는 0으로, Q는 1으로, ‘S’는 2로 변환해주세요. 모두 변환한 후에, 변환 결과를 head 함수를 이용해 확인해주세요. 
    1-3. SibSip(타이타닉호에 동승한 자매 및 배우자의 수), Parch(타이타닉호에 동승한 부모 및 자식의 수)를 통해서 family_size(가족크기)를 생성해주세요. 새로운 feature를 head 함수를 이용해 확인해주세요. 
</details>

#### 3. 모델 학습
- 담당 : 구수연
<details>
    <summary>
    Process
    </summary>
    
    1-1. 모델 학습 준비 : 이제 모델을 학습시키기 위한 데이터를 준비하겠습니다. 학습에 필요한 feature은 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', ‘family_size’ 입니다. feature과 target을 분리해주세요.  그 다음 데이터 스케일링을 진행해주세요.

    1-2. 이제 Logistic Regression, Random Forest, XGBoost를 통해서 생존자를 예측하는 모델을 학습하세요. 학습이 끝난 뒤 Logistic Regression과 Random Forest는 모델 accuracy를 통해, XGBoost는 mean squared error를 통해 test data를 예측하세요. 
</details>


