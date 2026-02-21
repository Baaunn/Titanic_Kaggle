## Overview 
This repository contains my solution for the Kaggle **Titanic** competition (tabular classification).
I built an end-to-end workflow with preprocessing, feature engineering, model selection, and ensembling.
Result: **Top 4.7% (681/14,671)** on the public leaderboard (**score: 0.79665**).
Key techniques: `Pipeline` + `ColumnTransformer`, `StratifiedKFoldCV`, `GridSearchCV`, and ensemble methods (Voting/Stacking).
Focus: improving generalization through leakage-aware validation and iterative feature engineering.

## Environment
Python (pandas, scikit-learn, xgboost).  
Notebooks are organized to follow the workflow from preprocessing → feature engineering → modeling.

# 캐글 타이타닉 생존자 예측 (Kaggle Titanic) 🚢🌊

## 1. 프로젝트 개요 (Overview)

* **목표:** 캐글의 "Titanic - Machine Learning from Disaster" 데이터를 활용하여, 탑승객의 정보를 기반으로 생존 여부를 예측하는 분류(Classification) 모델 개발
* **데이터:** 캐글 "Titanic - Machine Learning from Disaster" 대회 데이터 (링크: [Kaggle Competition Link](https://www.kaggle.com/c/titanic))
* **주요 특징:**
    * **캐글 리더보드 순위 681 / 14671 (상위 4.7%)** 달성
    * `Pipeline`과 `ColumnTransformer`를 활용한 체계적인 전처리 및 모델링 파이프라인 구축
    * `RandomForest`, `LogisticRegression`, `XGBoost` 등 개별 모델 튜닝 및 `Voting`, `Stacking` 앙상블 모델 비교

---

## 2. 사용 기술 스택 (Tech Stack) 🛠️

* **언어:** Python
* **라이브러리:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **주요 기법:** `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `StratifiedKFold`, `VotingClassifier`, `StackingClassifier`

---

## 3. 프로젝트 구조 (Directory Structure) 📂

* `data/`: 원본 데이터 (train.csv, test.csv) 및 전처리/피처 엔지니어링된 데이터 (df_all.csv, train_fe.csv, test_fe.csv) 저장
* `notebooks/`: 전처리, 피처 엔지니어링, 모델링 과정을 담은 Jupyter Notebooks
    * `preprocessing.ipynb` (데이터 전처리)
    * `feature_engineering.ipynb` (피처 엔지니어링)
    * `modeling.ipynb` (모델 학습, 튜닝 및 예측)
* `README.md`: 프로젝트 설명 파일

---

## 4. 분석 및 모델링 과정 (Workflow) 🚀

### 4.1 데이터 전처리 (`preprocessing.ipynb`)

* **데이터 통합:** `train`과 `test` 데이터를 통합하여 전처리 과정을 일관되게 적용
* **결측치(NaN) 처리:**
    * `Age`: 훈련 데이터의 중앙값(median)으로 대체
    * `Cabin`: 객실 번호의 첫 글자(Deck)만 추출하고, 결측치는 'Unknown'으로 대체
    * `Embarked`: 상관관계 분석을 통해 가장 가능성이 높은 'C'로 채움
    * `Fare`: Pclass와 Embarked가 동일한 다른 승객들의 요금 중앙값으로 대체하여 논리적인 추론을 통해 결측치를 채움

### 4.2 피처 엔지니어링 (`feature_engineering.ipynb`)

* **파생 변수 생성:**
    * `Title`: 이름(Name)에서 'Mr', 'Mrs', 'Miss' 등의 호칭을 추출하고, 희귀 호칭은 'Rare'로 통합
    * `FamilySize` & `FamilyCategory`: `SibSp`와 `Parch`를 조합하여 '가족 크기'를 계산하고, 이를 'Single', 'Small', 'Big'으로 범주화
    * `CabinGroup`: `Cabin` 정보를 'HighDeck', 'MidDeck', 'Unknown'으로 재그룹화
    * `AgeGroup` & `FareBinned`: `Age`와 `Fare`를 적절한 구간(bin)으로 나누어 범주형 변수로 만듦
* **상호작용 피처:**
    * `Age*Pclass`, `Sex_Pclass`: 생존율에 큰 영향을 미치는 두 변수의 상호작용 피처를 생성
    * `FarePerPerson`: `Fare`를 `FamilySize`로 나누어 '1인당 요금'을 계산
* **고급 피처 (Target Encoding):**
    * **주요 특징:** 데이터 누수(Data Leakage)를 방지하기 위해, 훈련/테스트 데이터에 **공통으로 존재하는** 'Family'와 'Ticket' 그룹을 식별
    * 이 공통 그룹의 생존율 중앙값을 `FamilySurvivalRate`, `TicketSurvivalRate`라는 피처로 생성하여 모델이 활용할 수 있도록 함

### 4.3 모델링 및 평가 (`modeling.ipynb`)

* **파이프라인 구축:** `ColumnTransformer`를 사용해 수치형 피처(`StandardScaler`)와 범주형 피처(`OneHotEncoder`) 처리를 자동화하고, 이를 `Pipeline`으로 모델과 통합
* **모델 비교 실험:**
    * `LogisticRegression` (L1, L2 규제 및 alpha 등 튜닝)
    * `RandomForestClassifier` (n_estimators, max_depth 등 튜닝)
    * `XGBClassifier` (learning_rate, max_depth 등 튜닝)
    * 모든 모델은 `GridSearchCV`와 `StratifiedKFold`를 사용하여 교차 검증 및 최적 하이퍼파라미터 탐색 수행
* **앙상블 모델:**
    * `VotingClassifier` (Soft Voting): 개별 모델(LR, RF, XGB)의 예측 확률을 가중 평균하여 최종 예측을 수행
    * `StackingClassifier`: 개별 모델을 1단계 예측기로, `LogisticRegression`을 메타(final) 모델로 사용하는 스태킹 앙상블을 구현
* **모델 선택:** `classification_report`, `confusion_matrix`와 **캐글 최종 점수**를 비교하여 일반화 성능이 우수한 **RandomForestClassifier** 모델 선택
* **모델 해석:** `RandomForest`와 `XGBoost`의 `feature_importances_` (피처 중요도), `LogisticRegression`의 `coef_` (계수)를 시각화하여, `SurvivalRate`, `Title`, `Sex` 등의 피처가 예측에 중요하게 작용함을 확인
      
---

## 5. 결과 및 해석 (Results & Interpretation)

* 개별 모델(RF, LR, XGB) 및 앙상블 모델(Voting, Stacking) 모두 검증 세트(Validation Set)에서 약 **85%**의 안정적인 정확도(Accuracy)를 보임
* 피처 중요도 분석 결과, `feature_engineering.ipynb`에서 생성한 `SurvivalRate` (가족/티켓 생존율) 피처가 `Title`이나 `Sex` 만큼이나 예측에 매우 강력한 영향을 미치는 핵심 피처임을 확인
* <img width="1233" height="672" alt="image" src="https://github.com/user-attachments/assets/3caa5db7-33fd-4694-b55c-8c16bfc54b5e" />
* 최종적으로 튜닝된 `RandomForestClassifier` 모델을 사용하여 캐글에 제출, Public Score 0.79665으로 **14,671팀 중 681등 (상위 4.7%)**의 성과를 달성

---

## 6. 향후 개선 방향 (Future Work) 💡

* **고급 인코딩:** `Cabin` 및 `Ticket` 피처에 대해 `CountEncoder` 외에 `TargetEncoder` 등을 직접 적용하여 성능 변화 관찰
* **이상치 분석:** `FarePerPerson`, `Age` 등의 피처에서 발견되는 이상치(outlier)가 모델에 미치는 영향을 분석하고 처리
