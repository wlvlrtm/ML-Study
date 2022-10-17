import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt


def main() :
    ## 데이터 준비
    fish = pd.read_csv("https://bit.ly/fish_csv_data")
    fish_input = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()
    fish_target = fish["Species"].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)


    ## 데이터 전처리
    ss = StandardScaler()
    ss.fit(train_input)
    train_scaled = ss.transform(train_input)
    test_scaled = ss.transform(test_input)


    ## 확률적 경사 하강법 적용
    sc = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
    sc.fit(train_scaled, train_target)
    print(sc.score(train_scaled, train_target))
    print(sc.score(test_scaled, test_target))


    ## 경사 하강; 부분 하강
    sc.partial_fit(train_scaled, train_target)
    print(sc.score(train_scaled, train_target))
    print(sc.score(test_scaled, test_target))


    ## 최적의 조기 종료 지점 탐색
    sc = SGDClassifier(loss="log_loss", random_state=42)
    train_score = []
    test_score = []
    classes = np.unique(train_target)
    for _ in range(0, 300) :
        sc.partial_fit(train_scaled, train_target, classes=classes)
        train_score.append(sc.score(train_scaled, train_target))
        test_score.append(sc.score(test_scaled, test_target))
    plt.plot(train_score)
    plt.plot(test_score)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


    ## 확률적 경사 하강법 적용; 최적의 조기 종료 지점 적용
    sc = SGDClassifier(loss="log_loss", max_iter=100, tol=None, random_state=42)
    sc.fit(train_scaled, train_target)

    
    ## 점수 출력
    print(sc.score(train_scaled, train_target))
    print(sc.score(test_scaled, test_target))



if __name__ == "__main__" :
    main()