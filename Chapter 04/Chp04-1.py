from pydoc import describe
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from scipy.special import softmax


## CSV 데이터 읽어오기 (Pandas)
fish = pd.read_csv("https://bit.ly/fish_csv_data")
fish.head()

## Species 열 출력 (CSV)
print(pd.unique(fish["Species"]))

## 타깃 데이터, 입력 데이터 지정 (입력 데이터 -> 타깃 데이터)
fish_input = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
fish_target = fish["Species"].to_numpy()

## 테스트 세트, 훈련 세트 분할
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

## 테스트 세트, 훈련 세트 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

## K-최근접 이웃 분류기 모델로 확률 예측
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

## 5개의 샘플 타깃 확률 출력
proba = kn.predict_proba(test_scaled[:5])
print(kn.predict(test_scaled[:5]))
print(kn.classes_)
print(np.round(proba, decimals=4))

## 4번째 샘플의 이웃 클래스 출력
_, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

## 시그모이드 함수, 그래프 테스트 (-5 ~ 5)
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

## 로지스틱 회귀 이진 분류
bream_smelt_indexes = (train_target == "Bream") | (train_target == "Smelt")
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

## 로지스틱 회귀 모델 훈련
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

## 학습한 계수 출력
print(lr.coef_, lr.intercept_)

## z 값 계산
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

## 시그모이드 함수로 확률 출력
print(expit(decisions))

## 로지스틱 회귀로 다중 분류 수행
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

## 5개 샘플에 대한 예측 출력
print(lr.predict(test_scaled[:5]))

## 예측 확률 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

## 7개의 생선 확률을 소프트맥스 함수로 계산
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))