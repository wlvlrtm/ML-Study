import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def main() :
    ## 데이터 준비
    df = pd.read_csv("https://bit.ly/perch_csv_data")
    perch_full = df.to_numpy()

    perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
        115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
        150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
        218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
        556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
        850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
        1000.0])


    ## 테스트 세트, 훈련 세트 분할
    train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)


    ## 특성 공학 적용
    poly = PolynomialFeatures()
    poly.fit([[2, 3]])
    print(poly.transform([[2, 3]]))


    ## 훈련 세트, 테스트 세트 변환
    poly = PolynomialFeatures(include_bias=False)
    poly.fit(train_input)
    train_poly = poly.transform(train_input)
    test_poly = poly.transform(test_input)


    ## 다항 회귀 점수 출력
    lr = LinearRegression()
    lr.fit(train_poly, train_target)


    ## 특성의 스케일 정규화
    ss = StandardScaler()
    ss.fit(train_poly)
    train_scaled = ss.transform(train_poly)
    test_scaled = ss.transform(test_poly)


    ## 릿지 규제 적용
    ridge = Ridge()
    ridge.fit(train_scaled, train_target)


    ## alpha 값 탐색하기
    train_score = []
    test_score = []

    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

    for alpha in alpha_list :
        ridge = Ridge(alpha = alpha)
        ridge.fit(train_scaled, train_target)
        train_score.append(ridge.score(train_scaled, train_target))
        test_score.append(ridge.score(test_scaled, test_target))

    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel("Alpha")
    plt.ylabel("R^2")
    plt.show()


    ## 릿지 규제 alpha 값 조정
    ridge = Ridge(alpha = 0.1)
    ridge.fit(train_scaled, train_target)


    ## 라쏘 규제 훈련
    lasso = Lasso()
    lasso.fit(train_scaled, train_target)


    ## 라쏘 규제 alpha 값 조정
    train_score = []
    test_score = []

    alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

    for alpha in alpha_list :
        lasso = Lasso(alpha = alpha, max_iter=10000)
        lasso.fit(train_scaled, train_target)
        train_score.append(lasso.score(train_scaled, train_target))
        test_score.append(lasso.score(test_scaled, test_target))

    plt.plot(np.log10(alpha_list), train_score)
    plt.plot(np.log10(alpha_list), test_score)
    plt.xlabel("Alpha")
    plt.ylabel("R^2")
    plt.show()


    ## 라쏘 규제 alpha 값 조정
    lasso = Lasso(alpha = 0.1)
    lasso.fit(train_scaled, train_target)


if __name__ == "__main__" :
    main()