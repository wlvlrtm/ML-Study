import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def main() :
    ## 데이터 준비
    wine = pd.read_csv("https://bit.ly/wine_csv_data")
    data = wine[["alcohol", "sugar", "pH"]].to_numpy()
    target = wine["class"].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)


    ## 랜덤 포레스트
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
    print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

    
    ## 랜덤 포레스트 특성 중요도 출력
    rf.fit(train_input, train_target)
    print(rf.feature_importances_)

    
    ## 랜덤 포레스트 OBB 출력
    rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
    rf.fit(train_input, train_target)
    print(rf.oob_score_)


    ## 엑스트라 트리
    et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
    scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
    print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))


    ## 엑스트라 트리 특성 중요도 출력
    et.fit(train_input, train_target)
    print(et.feature_importances_)


    ## 그라이디언트 부스팅 
    gb = GradientBoostingClassifier(random_state=42)
    scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
    print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))
    gb.fit(train_input, train_target)
    print(gb.feature_importances_)


    ## 히스토그램 기반 그레이디언트 부스팅
    hgb = HistGradientBoostingClassifier(random_state=42)
    scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
    print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))
    hgb.fit(train_input, train_target)
    result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
    print(result.importances_mean)
    result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
    print(result.importances_mean)
    hgb.score(test_input, test_target)


if __name__ == "__main__" :
    main()