import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def main() :
    ## 데이터 준비
    wine = pd.read_csv("https://bit.ly/wine_csv_data")
    

    ## 데이터 분리
    data = wine[["alcohol", "sugar", "pH"]].to_numpy()
    target = wine["class"].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)


    ## 데이터 전처리 진행
    ss = StandardScaler()
    ss.fit(train_input)
    train_scaled = ss.transform(train_input)
    test_scaled = ss.transform(test_input)


    ## 로지스틱 회귀 적용; 점수 출력 -> 불만족스러운 점수
    lr = LogisticRegression()
    lr.fit(train_scaled, train_target)
    print(lr.score(train_scaled, train_target))
    print(lr.score(test_scaled, test_target))


    ## 결정 트리 적용
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(train_scaled, train_target)
    print(dt.score(train_scaled, train_target))
    print(dt.score(test_scaled, test_target))


    ## 결정 트리 그리기; 전체
    plt.figure(figsize=(10, 7))
    plot_tree(dt)
    plt.show()

    
    ## 결정 트리 그리기; 부분
    plt.figure(figsize=(10, 7))
    plot_tree(dt, max_depth=1, filled=True, feature_names=["alcohol", "sugar", "pH"])
    plt.show()


    ## 결정 트리 훈련; 가지치기 = 3(최대 깊이); 전처리 적용
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(train_scaled, train_target)
    print(dt.score(train_scaled, train_target))
    print(dt.score(test_scaled, test_target))
    plt.figure(figsize=(20, 15))
    plot_tree(dt, filled=True, feature_names=["alcohol", "sugar", "pH"])
    plt.show()

    ## 결정 트리 훈련; 가지치기 = 3(최대 깊이); 전처리 미적용
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(train_input, train_target)
    print(dt.score(train_input, train_target))
    print(dt.score(test_input, test_target))
    plt.figure(figsize=(20, 15))
    plot_tree(dt, filled=True, feature_names=["alcohol", "sugar", "pH"])
    plt.show()



if __name__ == "_main_" :
    main()