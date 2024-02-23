# **📈 2024-1 ASCEND 채용 연계 데이터 분석 공모전**
![image](https://github.com/chasubeen/ASCEND/assets/98953721/47f281d2-bfb3-4ecc-8984-4d7b2715d5f7)


## **0. 대회 소개**
- 주어진 데이터를 활용하여 비트코인(BTC)의 단기 변동성 예측 모델링  
  ※ 모델은 Random Forest 만 사용!
    

### 주최/주관
- **주최/주관**: BDA (Big Data Analysis), ASCEND
- **협력 학회 및 동아리** : KUBIG, KHUDA, ESAA, ESC, PARROT

## **1. 참여자**
- 팀명: 코인채굴꾼
- **팀원 소개**
  
| 팀원 1 | 팀원 2 | 팀원 3 | 팀원 4 |
| --- | --- | --- | --- |
| <img src="https://github.com/chasubeen/Store-Sales-Forecasting/assets/98953721/3d28268b-f6a1-4935-ab53-08e2f79fe4e3" width = 100 height = 100> | <img src = "https://github.com/chasubeen/Store-Sales-Forecasting/assets/98953721/abe0ff1e-6df1-4863-bb9b-cd576aa7cc43" width = 100 height = 100> | <img src = "https://github.com/chasubeen/Store-Sales-Forecasting/assets/98953721/b7c82bb7-1880-4f33-af22-c062f993f945" width = 100 height = 100> | <img src = "https://avatars.githubusercontent.com/u/98953721?v=4" width = 100 height = 100> |
|[고도현](https://github.com/rhehgus02) | [박지현](https://github.com/Jihyun13579) | [원서현](https://github.com/seohyun126) | [차수빈](https://github.com/chasubeen) |

## **2. 진행 과정**

- **기간**: 2023.1.01(월) ~ 2023.01.19(금)
- **세부 일정**
  
| 날짜 | 내용 |
| --- | --- |
| 24/01/15 | 데이터 공개 |
| 24/01/21 | 1차 회의 <br> 데이터 확인 & 전처리 방향 고민|
| 24/01/28 | 2차 회의 <br> Feature Engineering 방식 논의 <br> 변동성 관련 지표 추가|
| 24/01/30 | 3차 회의 <br> 최종 활용 변수 선택 <br> 결측치 처리 방식 재논의 <br> 모델링 방식 논의|
| 24/02/01 | 4차 회의 <br> 최종 변수 선택 완료 <br> 모델링 방향 재논의|
| 24/02/03 | 최종 회의 <br> 데이터 가공 형태 최종 확인 → 수정사항 확인 후 재가공 <br> 예측을 위한 데이터 생성 <br> 모델 최종 학습 & 추론 과정 논의|
| 24/02/04 | 1차 과제물 제출 마감 |

- **역할**

| 이름 | 역할 |
| --- | --- |
| 고도현 | EDA, 모델링(모델 학습 & 추론) |
| 박지현 | 데이터 가공, EDA |
| 원서현 | EDA, Feature Engineering, 모델링(Baseline) |
| 차수빈 | 데이터 변환, 데이터 가공, EDA |

---

## **3. 과제 목표**
- 트레이딩에서 예측 오류는 손실로 직결될 가능성이 높기 때문에 예측 모델의 경우 과거의 학습된 데이터보다는 **아웃라이어가 많은 실제 시장 환경에서의 생존력이 더욱 중요**합니다.
  - 따라서 **제출된 모델은 향후 7일 동안의 ‘실제 시장 데이터’에 적용해 성능을 평가**받을 것입니다.
  <details>
  <summary>변동성의 중요성</summary>
  <pre>
  - 변동성(Volatility)은 트레이딩에서 가장 중요한 요소 중 하나 입니다.
  - 변동성을 이용해 수익을 창출하는 많은 종류의 전략들이 존재하고, 변동성 그 자체가 금융시장의 상황을 진단하는 도구가 되기도 합니다.
  - 특히 다양한 금융 상품의 현물/선물/옵션을 거래하는 ASCEND와 같은 퀀트 트레이딩 펌에서는 주어진 변동성을 이용하는 것 이외에도, 향후 변동성을 정확히 예측하는 것이 매우 중요합니다. 
  </pre>
  </details>
  
- 변동성과 금융 시장, 트레이딩에 대한 이해도 향상
- 방대한 양의 시계열 데이터 처리 능력 함양

### **📍 평가 지표**
- Mean Absolute Percentage Error(`MAPE`)
    - 실제값과 예측값 사이의 차이를 실제값으로 나눠줌으로써 오차가 실제값에서 차지하는 상대적인 비율을 산출하는 지표
    - 오차의 정도를 백분율 값으로 나타내기 때문에 모델의 성능을 직관적으로 이해하기 쉬우며, 타겟 변수가 여러개 일 때 각 변수별 모델의 성능을 평가하기 용이
- 다음과 같이 계산
    - $MAPE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |\frac{y_i-\hat{y}_i}{y_i}|$
        - $y_i$: 예측 대상인 실제 값
        - $\hat y_i$: 모델에 의한 예측 값
        - $n$: 시험 데이터셋의 크기
    - 예시
        
        ```python
        from sklearn.metrics import mean_absolute_percentage_error
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        mean_absolute_percentage_error(y_true, y_pred)
        
        y_true = [[0.5, 1], [-1, 1], [7, -6]]
        y_pred = [[0, 2], [-1, 2], [8, -5]]
        mean_absolute_percentage_error(y_true, y_pred)
        mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
        
        # the value when some element of the y_true is zero is arbitrarily high because
        # of the division by epsilon
        y_true = [1., 0., 2.4, 7.]
        y_pred = [1.2, 0.1, 2.4, 8.]
        mean_absolute_percentage_error(y_true, y_pred)
        ```
        

## **4. 디렉토리 구조**

```
2024 BDA 연합공모전_코인채굴꾼.
├─ 0. Parquet 데이터 다루기.ipynb 
├─ 1. 데이터 가공.ipynb 
├─ 2. EDA & Feature Engineering.ipynb 
└─ 3. Modeling.ipynb 
```

## **5. Data Description**

### 데이터 형태 변경

- 효율적인 메모리 활용을 위해 `.csv` 파일 형태로 제공된 원본 데이터를 `.parquet` 데이터로 재가공

### 원본 데이터

- `.parquet` 형태로 가공한 파일을 읽어와 기본 정보 확인
    - 결측치 등
- Binance 비트코인 거래소의 tick data

| Column Name | Description | Example |
| --- | --- | --- |
| id |각 거래에 고유하게 할당된 식별 번호 <br> 특정 거래를 구별하는데 사용함 | 4241445555 |
| price |거래가 체결된 가격 <br> 해당 틱에서 비트코인이 거래된 달러 가치를 나타냄 | 35000.0 |
| qty | 거래된 비트코인의 양 | 3 |
| quote_qty |거래된 비트코인의 달러 가치 합계 <br> price 와 qty를 곱한 값으로, 거래의 규모를 나타냄 | 105000 |
| time |거래가 기록된 유닉스 타임스탬프 시간 | 1696118759240 |
| is_buyer_maker |거래에서 구매자가 maker 인지 아닌지 나타내는 boolean 값 <br> True 이면 구매자가 maker, False 이면 구매자가 taker <br> *maker : 시장에서 신규 주문을 제공하여 유동성을 만드는 사람 <br> *taker : 시장 가격에 맞춰 즉시 거래를 체결한 사람| TRUE |


- 이후 아래의 코드를 활용하여  1시간 단위의 `OHLCV` 형태로 변형함
    
    ```python
    def convert_tick_to_ohlcv(data):
        """
        Converts given Binance tick data into 1-hour interval OHLCV (Open, High, Low, Close, Volume) data.
        :param data: DataFrame with Tick data
        :return: DataFrame with the Open, High, Low, Close, Volume values
        """
    
        data['time'] = pd.to_datetime(data['time'], unit='ms')
        ohlcv = data.resample('1H', on='time').agg({
            'price': ['first', 'max', 'min', 'last'],
            'qty': 'sum'
    })
    
        ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return ohlcv
    ```
    

### OHLCV 데이터

| Column Name | Description | Example |
| --- | --- | --- |
| Time(index) |1시간마다의 데이터를 하나로 묶어 OHLCV를 추출해냈을 때, 해당 행 데이터의 시간을 나타냄 | 2023-01-01 00:00:00 |
| Open |해당 시간 간격 동안의 최고 거래 가격 | 16537.5 |
| High |거래된 비트코인의 양 | 16540.9 |
| Low |해당 시간 간격 동안의 최저 거래 가격 | 16504.0 |
| Close |해당 시간 간격 동안의 마지막 거래 가격 | 16527.0 |
| Volume |해당 시간 간격 동안의 거래량 합계(거래된 비트코인 양의 합계) | 5381.399 |

- OHLCV 데이터 가공 이후 발생하는 결측치 처리
    - 특정 시간에 거래가 발생하지 않은 경우 결측치 발생
    - 해당 시간에 거래가 실제로 발생하지 않은 것이 아닌, 기록 누락으로 판단 후 결측치 보간
    - 선형 보간법 활용 ⇒  `interpolate(method = 'time')`
- 가공된 데이터는 `OHLCV.csv` 파일로 저장

## **6. EDA & Feature Engineering**

- 가공된 OHLCV 데이터를 불러와 간단한 EDA 진행
    - 캔들차트, 시계열 분석(시도표, 시계열분해, 정상성 검정)
- 이후 `Open`, `High`, `Low`, `Close`에 대해 ARIMA 모델링과 Prophet 패키지를 활용하여 예측치 생성
    - 2024/01/28 0:00 ~ 2024/01/31 0:00까지 1시간 단위로 예측
- `Volume`의 경우 2024/01의 시간대별 평균치로 대체
- Open, High, Low, Close, Volume을 활용하여 기술적 지표들을 추가
    - `ta` 라이브러리 활용
    - 변동성 지표, 추세 지표, 모멘텀 지표, 거래량 지표
- 최종적으로 가공된 데이터는 train과 test로 분리
    - train: 2023/01/01 0:00 ~ 2024/01/27 23:00(1시간 단위)
    - test: 2024/01/28 0:00 ~ 2024/01/31 0:00(1시간 단위)

## **7. 모델링**

- `train.csv`와 `test.csv`를 불러와 기본 정보 확인
- Feature 변수와 Target 변수 분리 후, 스케일링 진행
    - StandardScaler 활용
    - feature 변수에 대해서만 scaling
- Feature selection 진행
    - RandomForestRegressor Baseline 모델을 학습시킨 후 Feature Importance 확인
    - RFECV(Recursive Feature Elimination with Cross-Validation) 방식을 활용하여 최종적으로 활용할 변수 선택
- 피처 중요도와 RFECV를 통해 최종적으로 10개의 feature 변수를 선택 후 모델링
    
    ![image](https://github.com/chasubeen/ASCEND/assets/98953721/f6fa4ac8-b1d3-4a2e-8e67-4bd588e61280)

    - `ATR`, `UI`, `KST`, `DCL`, `PPO`, `MI`, `TRIX`, `VPT`, `FI`, `BLB`
- 시계열 교차 검증 방식을 활용하여 파라미터 튜닝(모델 최적화) 진행
    - TimeSeriesSplit(n_splits = 5)

## **8. 예측 & 결과 정리**

- 모델링 후 최종적으로 2024/01/28 0:00 ~ 2024/01/31 0:00에 대한 volatility를 예측
- 아래 코드를 활용하여 변동성 계산
    
    ```python
    def calculate_volatility(data, window=20):
        """
        Calculate the rolling volatility using the standard deviation of returns.
        :param data: DataFrame with OHLCV data
        :param window: The number of periods to use for calculating the standard deviation
        :return: DataFrame with the volatility values
        """
    
        # Calculate daily returns
        data['returns'] = data['Close'].pct_change()
    
        # Calculate the rolling standard deviation of returns
        data['volatility'] = data['returns'].rolling(window=window).std()
    
    return data
    ```
    
- 최적 파라미터
    
    ```python
    'max_depth': None, 
    'min_samples_leaf': 2, 
    'min_samples_split': 5, 
    'n_estimators': 200
    ```
    
- 최적 MAPE: 0.2319

---

## **회고**

### 고도현

- 경제 관련 시계열 데이터를 처음 다루어보기 때문에 경제 관련 사전 정보를 학습하는 데에 많은 시간을 들였다. 생소한 개념들이라 어렵기도 했지만 확실히 데이터를 이해하는 데에 도움이 되었고, 다른 금융 데이터를 다룰 때도 큰 도움이 될 것 같다.
- 마지막 target 변수를 예측하는 법을 두 가지 방법으로 나누어 생각했었는데, 주최 측의 가이드라인에 따라 우리의 생각과는 다른 방법을 채택했다. 다른 금융 데이터를 다루게 된다면 본래 생각했던 방법도 사용해보고 싶다.
- 예측해야 하는 기간의 칼럼 변수들을 모두 예측해야 했고 valid 데이터를 만들지 못했기 때문에 우리가 만든 모델의 성능을 정확하게 파악하지 못했다. 정답이 주어진다면 성능을 보고 다른 보완점을 찾아보고 싶다.

### 박지현

- 

### 원서현
- 본 데이터분석 대회를 통해 금융 데이터를 처음 접해보며 여러 금융 용어에 어려움이 있었으나 이번 기회를 통해 해당 도메인에 대한 배경지식을 공부할 수 있었던 좋은 경험이었다.
- test 데이터에 target 변수만 있는 상태에서 여러 feature 변수에 train 데이터를 통한 예측치를 넣고 이를 통한 modeling을 통해 target 변수를 예측하였다.
- test 데이터에 feature 변수를 어떻게 처리해야 할지 많은 고민을 했었던 경험이 매우 유익하였다. 

### **차수빈**

- 실제 기업에서 다루는 금융 데이터를 활용하여 여러 분석을 해보며 금융 도메인에 대한 지식을 쌓을 수 있어서 좋았다.
- 또한, 시계열적 특성을 모델링에 반영하기 위해 여러 기술들을 찾아보며 시계열 데이터 분석에 대해 더 깊게 공부할 수 있었고, 또한 변수가 많은 경우 RFECV 등의 변수 선택 방법을 활용할 수 있다는 점도 새롭게 알게 되었다.
- 처음 보는 금융/투자 관련 용어들과 지표들이 많아 데이터를 이해하는 것에 상당한 어려움이 있었고, 그로 인해 프로젝트 전체 진행 과정이 매끄럽지 못하였던 것 같아 아쉽다.
