# 엔지니어를 위한 케라스 소개

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2020/04/01<br>
**최근 변경:** 2020/04/28<br>
**설명:** 케라스를 써서 실용적인 기계 학습 솔루션을 만들기 위해 필요한 모든 것.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/intro_to_keras_for_engineers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/intro_to_keras_for_engineers.py)



---
## 준비


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
```

---
## 들어가며

케라스를 이용해 실제 제품에 딥 러닝 기반 기능을 넣으려고 하는
기계 학습 엔지니어라면 이 안내서에서 케라스 API의 핵심 개념들을
접할 수 있다.

이 안내서에서 다음을 배우게 된다.

- 모델 훈련 전에 데이터 준비하기.
  (NumPy 배열이나 `tf.data.Dataset` 객체로 바꾸기)
- 데이터 전처리하기. 가령 피처 정규화나 어휘 색인 만들기.
- 데이터를 유용한 예측으로 바꿔 주는 모델을
  케라스 함수형 API로 만들기.
- 케라스에 내장된 `fit()` 메서드로 모델 훈련시키기.
  체크포인트 저장, 지표 관찰, 장애 내성까지.
- 테스트용 데이터로 모델 평가하기. 그 모델을 가지고 새 데이터로 추론하는 방법.
- `fit()`의 동작 방식 바꾸기. 가령 GAN 만들기.
- 여러 GPU 활용해서 훈련 속도 높이기.
- 하이퍼파라미터 조정을 통해 모델 개선하기.

이 개념들을 견고하게 해 주는 다음 전구간 예시들에 대한 링크를
안내서 끝에서 볼 수 있다.

- 이미지 분류
- 텍스트 분류
- 신용카드 사기 탐지


---
## 데이터 적재 및 전처리하기

신경망은 텍스트 파일이나 JPEG 형식 이미지 파일, CSV 파일 같은 비가공 데이터를
처리하지 않는다. **벡터화** 및 **표준화**한 표현을 처리한다.

- 텍스트 파일은 문자열 텐서들로 읽어들인 후 단어들로 쪼개야 한다. 마지막으로
 그 단어들의 색인을 만들어서 정수 텐서들로 바꿔야 한다.
- 이미지는 읽어서 정수 텐서들로 디코딩해야 한다. 그 다음 부동소수점으로 변환하고
 작은 값(일반적으로 0과 1 사이)으로 정규화한다.
- CSV 데이터는 파싱해서 수치 피처는 부동소수점 텐서로 변환해야 하고
 범주형 피처는 색인해서 정수 텐서로 변환해야 한다.
 그러고 나서 보통 각 피처를 평균 0에 분산이 단위값이 되도록 정규화해야 한다.
- 기타 등등.

데이터 적재부터 시작해 보자.

---
## 데이터 적재하기

케라스 모델은 세 가지 입력을 받는다.

- **NumPy 배열**: Scikit-Learn 및 기타 파이썬 기반 라이브러리들과 동일. 데이터가
 메모리에 들어가는 경우 좋은 방식이다.
- **[텐서플로 `Dataset` 객체](https://www.tensorflow.org/guide/data)**: 메모리에
 들어가지 않으며 디스크나 분산 파일 시스템에서 스트림으로 가져오는 데이터셋에
 적합한 고성능 방식이다.
- **파이썬 제너레이터**: (`keras.utils.Sequence`의 서브클래스 같은) 데이터 배치들을
 내놓는 제너레이터.

모델 훈련을 시작하기 전에 데이터를 이 형식들 중 하나로 만들어야 할 것이다.
데이터셋이 크고 GPU(들)에서 훈련시키려 한다면 `Dataset` 객체 사용을 고려하자.
성능에 큰 영향을 끼치는 다음과 같은 세부 동작들을 해 준다.

- GPU가 바쁜 동안 CPU에서 비동기적으로 데이터를 전처리해서 큐에 넣어 두기.
- GPU가 앞선 배치 처리를 마쳤을 때 바로 이용할 수 있도록 GPU 메모리에 데이터를
 미리 올려서 GPU를 완전히 활용할 수 있게 하기.

케라스에는 디스크의 데이터를 `Dataset`으로 바꾸는 걸 도와 주는 다양한 유틸리티가 있다.

- `tf.keras.preprocessing.image_dataset_from_directory`는 분류별 폴더에 나눠져
 있는 이미지 파일들을 레이블 붙은 이미지 텐서 데이터셋으로 바꿔 준다.
- `tf.keras.preprocessing.text_dataset_from_directory`는 텍스트 파일에 대해 같은
 동작을 해 준다.

또한 CSV 파일에서 구조화된 데이터를 적재하는 `tf.data.experimental.make_csv_dataset`
같은 비슷한 유틸리티들이 텐서플로의 `tf.data`에 포함돼 있다.

**예: 디스크의 이미지 파일들에서 레이블 붙은 데이터셋 얻기**

다음처럼 분류에 따라 여러 폴더에 나눠져 있는 이미지 파일들이 있다고 하자.

```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

다음처럼 할 수 있다.

```python
# 데이터셋 만들기.
dataset = keras.preprocessing.image_dataset_from_directory(
  'path/to/main_directory', batch_size=64, image_size=(200, 200))

# 확인을 위해 데이터셋이 내놓은 배치들을 순회해 보기.
for data, labels in dataset:
   print(data.shape)  # (64, 200, 200, 3)
   print(data.dtype)  # float32
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```

알파벳 순서로 폴더의 순번이 표본의 레이블이 된다. 물론 예를 들어
`class_names=['class_a', 'class_b']`라고 따로 설정할 수도 있다. 이렇게 하면
`class_a`가 레이블 `0`이 되고 `class_b`가 레이블 `1`이 된다.

**예: 디스크의 텍스트 파일에서 레이블 붙은 데이터셋 얻기**

텍스트도 비슷하다. 분류에 따라 여러 폴더에 나눠져 있는 `.txt` 문서들이 있다면
다음처럼 할 수 있다.

```python
dataset = keras.preprocessing.text_dataset_from_directory(
  'path/to/main_directory', batch_size=64)

# 확인을 위해 데이터셋이 내놓은 배치들을 순회해 보기.
for data, labels in dataset:
   print(data.shape)  # (64,)
   print(data.dtype)  # string
   print(labels.shape)  # (64,)
   print(labels.dtype)  # int32
```



---
## 케라스로 데이터 전처리하기

데이터가 str/int/float NumPy 배열 형태 내지 str/int/float 텐서 배치들을 내놓는
`Dataset` 객체(또는 파이썬 제너레이터)라면 이제 데이터를 **전처리**할 차례다.
다음이 포함될 수 있다.

- 문자열 데이터를 토큰들로 나누고 토큰 색인화.
- 피처 정규화.
- 데이터 값을 작게 줄이기. 일반적으로 신경망의 입력 값은 0에 가까워야 하며 보통
 평균 0에 분산이 단위값인 데이터나 범위가 `[0, 1]`인 데이터를 기대한다.

### 완벽한 기계 학습 모델은 전구간을 처리한다

가능하면 외부의 데이터 전처리 파이프라인을 통해서가 아니라 **모델의 일부로서**
데이터 전처리를 하는 게 일반적으로 좋다. 데이터 전처리를 외부에서 하면 모델을
실제 사용할 때 이식성이 떨어지기 때문이다. 가령 텍스트를 처리하는 어느 모델을
생각해 보자. 그 모델에선 특정 토큰화 알고리듬과 특정 어휘 색인을 사용한다.
모델을 모바일 앱이나 자바스크립트 앱으로 만들려고 할 때 똑같은 전처리 구성을
그 언어로 다시 만들어야 할 것이다. 간단한 일이 아닐 수 있다. 원래 파이프라인과
다시 만든 버전 사이에 작은 차이만 있어도 모델이 완전히 무효화되거나 적어도
성능이 심각하게 떨어질 가능성이 있다.

이미 전처리가 포함돼 있는 전구간 모델을 내놓을 수 있다면 일이 훨씬 수월할 것이다.
**모델에서 가급적 비가공 데이터에 가까운 뭔가를 입력으로 기대하는 게 바람직하다.
이미지 모델이라면 `[0, 255]` 범위 RGB 픽셀 값들을 기대해야 하고, 텍스트 모델이라면
`utf-8` 문자열들을 받아들여야 한다.** 그러면 그 모델 소비자는 전처리 파이프라인에
대해 생각할 필요가 없다.

### 케라스 전처리 층 사용하기

케라스에선 다음과 같은 **전처리 층**을 통해 모델 내 데이터 전처리를 한다.

- `TextVectorization` 층을 통해 텍스트의 문자열 벡터화하기
- `Normalization` 층을 통해 피처 정규화하기
- 이미지 크기 조정, 자르기, 이미 데이터 증강

케라스 전처리 층들의 주된 장점은 훈련 중이나 훈련 후에 **모델에 직접 포함시켜서**
모델을 이식성 있게 만들 수 있다는 점이다.

일부 전처리 층에는 상태가 있다.

- `TextVectorization`은 단어 내지 토큰을 정수 색인으로 매핑하는 색인을 가지고 있다.
- `Normalization`은 피처의 평균 및 분산을 가지고 있다.

훈련 데이터의 표본 하나(또는 전체)로 `layer.adapt(data)`를 호출하면 전처리 층의
상태를 얻을 수 있다.


**예: 문자열들을 정수 단어 색인 열로 바꾸기**



```python
from tensorflow.keras.layers import TextVectorization

# dtype이 `string`인 예시 훈련 데이터
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# TextVectorization 층 인스턴스 만들기. 정수 토큰 색인들을 반환하도록,
# 또는 조밀한 토큰 표현(가령 멀티핫이나 TF-IDF)을 반환하도록 설정할 수 있다.
# 텍스트 표준화 알고리듬과 텍스트 분할 알고리듬도 설정 가능하다.
vectorizer = TextVectorization(output_mode="int")

# 배열이나 데이터셋으로 층의 `adapt`를 호출하면 그 데이터에 대한 어휘 색인을
# 만든다. 이후 새 데이터에 그 색인을 재사용할 수 있다.
vectorizer.adapt(training_data)

# adapt 호출을 하고 나면 앞서 `adapt()` 데이터에서 본 어떤 n그램이든 인코딩할
# 수 있다. 모르는 n그램은 "out-of-vocalbulary" 토큰으로 인코딩한다.
integer_data = vectorizer(training_data)
print(integer_data)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[4 5 2 9 3]
 [7 6 2 8 3]], shape=(2, 5), dtype=int64)

```
</div>
**예: 문자열들을 원샷 인코딩 바이그램 열로 바꾸기**


```python
from tensorflow.keras.layers import TextVectorization

# dtype이 `string`인 예시 훈련 데이터
training_data = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])

# TextVectorization 층 인스턴스 만들기. 정수 토큰 색인들을 반환하도록,
# 또는 조밀한 토큰 표현(가령 멀티핫이나 TF-IDF)을 반환하도록 설정할 수 있다.
# 텍스트 표준화 알고리듬과 텍스트 분할 알고리듬도 설정 가능하다.
vectorizer = TextVectorization(output_mode="binary", ngrams=2)

# 배열이나 데이터셋으로 층의 `adapt`를 호출하면 그 데이터에 대한 어휘 색인을
# 만든다. 이후 새 데이터에 그 색인을 재사용할 수 있다.
vectorizer.adapt(training_data)

# adapt 호출을 하고 나면 앞서 `adapt()` 데이터에서 본 어떤 n그램이든 인코딩할
# 수 있다. 모르는 n그램은 "out-of-vocalbulary" 토큰으로 인코딩한다.
integer_data = vectorizer(training_data)
print(integer_data)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[0. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1.]
 [0. 1. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0.]], shape=(2, 17), dtype=float32)

```
</div>
**예: 피처 정규화하기**


```python
from tensorflow.keras.layers import Normalization

# [0, 255] 범위 값의 예시 이미지 데이터
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)
print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))
```

<div class="k-default-codeblock">
```
var: 1.0000
mean: -0.0000

```
</div>
**예: 이미지 값 크기 조정하고 가운데 잘라 남기기**

`Rescaling` 층과 `CenterCrop` 층 모두 상태가 없으므로 이 경우엔 `adapt()`를 호출할
필요가 없다.


```python
from tensorflow.keras.layers import CenterCrop
from tensorflow.keras.layers import Rescaling

# [0, 255] 범위 값의 예시 이미지 데이터
training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

cropper = CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0 / 255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))
```

<div class="k-default-codeblock">
```
shape: (64, 150, 150, 3)
min: 0.0
max: 1.0

```
</div>
---
## 케라스 함수형 API로 모델 만들기

"층"이란 (위의 값 크기 조정 및 가운데 잘라 남기기 변환 같은) 단순한 입력-출력
변환이다. 예를 들어 다음은 입력들을 16차원 피처 공간으로 매핑해 주는 선형 투사
층이다.

```python
dense = keras.layers.Dense(units=16)
```

"모델"이란 층들로 이뤄진 방향 있는 무순환 그래프다. 여러 하위 층들을 포함하며
데이터에 노출시켜 훈련시킬 수 있는 "큰 층"으로 생각할 수도 있다.

케라스 모델을 만드는 가장 일반적이고 가장 강력한 방법이 함수형 API다.
함수형 API로 모델을 만들려면 먼저 입력 모양을 (그리고 선택적으로 dtype을)
지정해 줘야 한다. 입력 중 어느 차원이 가변적이라면 `None`으로 지정하면 된다.
예를 들어 200x200짜리 RGB 이미지 입력은 `(200, 200, 3)` 모양이 될 테고
임의 크기의 RGB 이미지 입력은 `(None, None, 3)` 모양이 된다.


```python
# 입력으로 임의 크기 RGB 이미지들을 받는다고 하자
inputs = keras.Input(shape=(None, None, 3))
```

입력(들)을 지정한 다음엔 그 위로 층 변환들을 차례로 연결해서 최총 출력까지
연결할 수 있다.


```python
from tensorflow.keras import layers

# 이미지 가운데 150x150 잘라 남기기
x = CenterCrop(height=150, width=150)(inputs)
# 이미지 값 크기를 [0, 1]로 조정하기
x = Rescaling(scale=1.0 / 255)(x)

# 합성곱 층과 풀링 층 적용
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(3, 3))(x)
x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)

# 전역 평균 풀링 적용해서 평면 피처 벡터 얻기
x = layers.GlobalAveragePooling2D()(x)

# 가장 위에 조밀 분류자 추가
num_classes = 10
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

입력(들)을 출력으로 바꿔 주는 방향 있는 무순환 층 그래프를 정의했으면
`Model` 객체를 만들자.


```python
model = keras.Model(inputs=inputs, outputs=outputs)
```

이 모델은 기본적으로 큰 층처럼 동작한다. 다음처럼 데이터 배치들을 가지고
호출할 수 있다.


```python
data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")
processed_data = model(data)
print(processed_data.shape)
```

<div class="k-default-codeblock">
```
(64, 10)

```
</div>
모델 각 단계에서 데이터가 어떤 모양으로 변환되는지를 간략히 찍을 수 있다.
디버깅에 유용하다.

층별로 표시되는 출력 모양에는 **배치 크기**도 포함된다는 점에 유의하자.
여기선 배치 크기가 None인데, 모델이 임의 크기의 배치를 처리할 수 있다는 뜻이다.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, None, None, 3)]   0         
_________________________________________________________________
center_crop_1 (CenterCrop)   (None, 150, 150, 3)       0         
_________________________________________________________________
rescaling_1 (Rescaling)      (None, 150, 150, 3)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 49, 49, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 47, 47, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 32)        9248      
_________________________________________________________________
global_average_pooling2d (Gl (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 10)                330       
=================================================================
Total params: 19,722
Trainable params: 19,722
Non-trainable params: 0
_________________________________________________________________

```
</div>
함수형 API를 이용하면 입력이 여럿(예를 들어 이미지와 그 메타데이터)이거나
출력이 여럿(예를 들어 이미지 분류 및 사용자가 클릭할 가능성을 예층)인
모델을 만드는 게 쉬워진다. 어떤 게 가능한지 자세히 알고 싶으면 [함수형 API
안내서](/guides/functional_api/)를 보라.

---
## `fit()`으로 모델 훈련시키기

이제 다음을 알게 됐다.

- 데이터를 (가령 NumPy 배열이나 `tf.data.Dataset` 객체로) 준비하는 방법
- 데이터를 처리할 모델을 만드는 방법

다음은 데이터로 모델을 훈련시킬 차례다. `Model` 클래스에는 훈련 루프인
`fit()` 메서드가 내장돼 있다. `Dataset` 객체, 데이터 배치를 내놓는
제너레이터, NumPy 배열을 받을 수 있다.

`fit()`을 호출하기 전에 먼저 최적화 기법과 손실 함수를 (이 개념들에 이미
익숙하다고 가정한다.) 지정해야 한다. 바로 `compile()` 단계다.

```python
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy())
```

문자열 식별자로 손실 함수와 최적화 기법을 지정할 수도 있다. (이 경우
생성자 기본 인자 값들을 쓴다.)


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

모델을 컴파일하고 나면 모델을 데이터에 "맞출(fit)" 수 있게 된다.
NumPy 데이터로는 다음과 같이 모델을 맞추게 된다.

```python
model.fit(numpy_array_of_samples, numpy_array_of_labels,
          batch_size=32, epochs=10)
```

데이터 외에도 두 가지 핵심 매개변수를 지정해 줘야 한다. `batch_size`와
에포크 수(데이터 반복 횟수)다. 위와 같이 하면 데이터를 표본 32개짜리 배치들로
나누고 훈련 동안 모델이 데이터를 10번 돌게 된다.

데이터셋으로는 다음과 같이 모델을 맞추게 된다.

```python
model.fit(dataset_of_samples_and_labels, epochs=10)
```

데이터셋이 내놓는 데이터는 이미 배치들로 나눠져 있을 것이기에 배치
크기를 지정할 필요가 없다.

MNIST 숫자 분류법을 학습하는 모형 모델을 가지고 전체적으로 어떻게
도는지를 보자.


```python
# Numpy 배열로 데이터 얻기
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 간단한 모델 만들기
inputs = keras.Input(shape=(28, 28))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# 모델 컴파일하기
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Numpy 데이터로 1 에포크만큼 모델 훈련시키기
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# 데이터셋으로 1 에포크만큼 모델 훈련시키기
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
rescaling_2 (Rescaling)      (None, 28, 28)            0         
_________________________________________________________________
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               100480    
_________________________________________________________________
dense_2 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                1290      
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
_________________________________________________________________
Fit on NumPy data
938/938 [==============================] - 1s 940us/step - loss: 0.4771
Fit on Dataset
938/938 [==============================] - 1s 942us/step - loss: 0.1138

```
</div>
`fit()` 호출은 훈련 과정 동안 어떤 일이 일어났는지 기록한 "이력" 객체를 반환한다.
`history.history` 딕셔너리에는 지표들의 에포크 단위 시계열 값이 담겨 있다.
(이번에는 지표가 손실 값 하나뿐이고 에포크도 한 번이므로 스칼라 값 하나만 있다.)


```python
print(history.history)
```

<div class="k-default-codeblock">
```
{'loss': [0.11384169012308121]}

```
</div>
[케라스 내장 메서드를 이용한 훈련과 평가 안내서](
  /guides/training_with_built_in_methods/)에서 자세한 `fit()` 사용 방법을 볼 수 있다.

### 성능 지표 추적하기

모델을 훈련시키다 보면 분류 정확도, 정밀도, 재현율, AUC 등 지표들을 추적하고 싶어진다.
또한 훈련 데이터뿐 아니라 검사용 데이터셋에 대해서도 그 지표들을 관찰하고 싶다.

**지표 관찰하기**

다음처럼 지표 객체 목록을 `compile()`에 줄 수 있다.



```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
history = model.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
938/938 [==============================] - 1s 929us/step - loss: 0.0835 - acc: 0.9748

```
</div>
**`fit()`에 검사용 데이터 주기**

`fit()`에 검사용 데이터를 줘서 검사 손실 및 검사 지표들을 관찰할 수 있다. 각 에포크가
끝날 때마다 검사 지표들을 보고한다.


```python
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs=1, validation_data=val_dataset)
```

<div class="k-default-codeblock">
```
938/938 [==============================] - 1s 1ms/step - loss: 0.0563 - acc: 0.9829 - val_loss: 0.1041 - val_acc: 0.9692

```
</div>
### 체크포인트 등을 위해 콜백 사용하기

훈련 시간이 몇 분을 넘어간다면 도중에 정기적으로 모델을 저장할 필요가 있다.
혹시라도 훈련 과정이 비정상 종료되는 경우에 저장된 모델을 이용해 훈련을
재개할 수 있다. (다중 작업자 분산 훈련에서 중요하다. 작업 장비가 많으면
적어도 하나는 언젠가 문제가 생기기 마련이다.)

케라스의 중요한 기능으로 `fit()`에 설정하는 **콜백**이 있다. 콜백이란
훈련 중 다음과 같은 여러 시점에 모델에서 호출하는 객체다.

- 각 배치 시작과 끝
- 각 에포크 시작과 끝

콜백을 이용하면 모델 훈련 전체의 각본을 만들 수 있다.

콜백을 이용해 모델을 주기적으로 저장할 수 있다. 다음이 간단한 예인데,
`ModelCheckpoint` 콜백을 설정해서 각 에포크 끝에서 모델을 저장하게 한다.
파일 이름에 현재 에포크 번호가 들어가게 된다.

```python
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='path/to/my/model_{epoch}',
        save_freq='epoch')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

콜백을 이용해 주기적으로 최적화 학습률을 바꾸거나 슬랙 봇으로 지표를 보내거나
훈련 종료 시 이메일 알림을 보내는 것 등을 할 수도 있다.

이용 가능한 콜백 종류와 자체 콜백 작성법에 대해선 [콜백 API 문서](/api/callbacks/)와
[자체 콜백 작성하기 안내서](/guides/writing_your_own_callbacks/)를 볼 수 있다.

### 텐서보드로 훈련 진행 상황 관찰하기

케라스의 진행 막대를 쳐다보는 게 손실과 지표들의 변화를 관찰하는 가장 편안한
방식은 아니다. 더 나은 방법이 있는데, 지표들을 실시간 그래프로 표시할 수 있는
웹 응용인 [텐서보드](https://www.tensorflow.org/tensorboard)다.

`fit()`에서 텐서보드를 이용하려면 텐서보드 로그를 저장할 디렉터리를 지정한
`keras.callbacks.TensorBoard` 콜백을 인자로 주기만 하면 된다.


```python
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=2, callbacks=callbacks)
```

그러고 나면 브라우저에서 텐서보드 인스턴스를 띄워서 그 위치에 기록되는 로그들을
관찰할 수 있다.

```
tensorboard --logdir=./logs
```

추가로 Jupyter / Colab 노트북에서 모델을 훈련시킬 때는 인라인 텐서보드 탭을
띄울 수도 있다.
[여기 자세한 설명이 있다](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).

### `fit()` 완료 후: 새 데이터로 검사 성능 평가하고 예측 생성하기

훈련을 마친 모델이 있으면 `evaluate()`을 통해 새 데이터에 대한 손실과 지표들을
평가할 수 있다.


```python
loss, acc = model.evaluate(val_dataset)  # 손실 및 지표 반환
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)
```

<div class="k-default-codeblock">
```
157/157 [==============================] - 0s 688us/step - loss: 0.1041 - acc: 0.9692
loss: 0.10
acc: 0.97

```
</div>
`predict()`를 통해 NumPy 배열로 된 예측(모델 출력 층(들)의 활성 값)을
생성할 수도 있다.


```python
predictions = model.predict(val_dataset)
print(predictions.shape)
```

<div class="k-default-codeblock">
```
(10000, 10)

```
</div>
---
## 자체 훈련 단계로 `fit()` 사용하기

기본적으로 `fit()`은 **지도 학습**을 하게 구성돼 있다. 다른 종류의 훈련
루프(예를 들어 GAN 훈련 루프)가 필요하다면 `Model.train_step()` 메서드를
자체적으로 구현할 수 있다. `fit()` 수행 동안 반복해서 호출되는 메서드다.

지표, 콜백 등은 이전 그대로 동작한다.

다음은 `fit()`이 원래 하는 동작을 재구현한 간단한 예시다.

```python
class CustomModel(keras.Model):
  def train_step(self, data):
    # 데이터 풀기. 모델과 `fit()` 인자에 따라 그 구조가 달라진다.
    x, y = data
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)  # 진행
      # 손실 값 계산
      # (손실 함수는 `compile()`로 설정)
      loss = self.compiled_loss(y, y_pred,
                                regularization_losses=self.losses)
    # 경사 계산
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    # 가중치 갱신
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # 지표 갱신 (손실 추적용 지표 포함)
    self.compiled_metrics.update_state(y, y_pred)
    # 지표 이름으로 현재 값 얻을 수 있는 딕셔너리 반환
    return {m.name: m.result() for m in self.metrics}

# CustomModel 인스턴스 구성 및 컴파일
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=[...])

# 하던 대로 `fit` 이용
model.fit(dataset, epochs=3, callbacks=...)
```

내장 훈련 루프와 평가 루프의 동작 방식을 바꾸는 방법에 대한 자세한 내용은
[`fit()` 내부 동작 바꾸기](/guides/customizing_what_happens_in_fit/)를 보라.

---
## 열심 실행 방식으로 모델 디버깅하기

자체적인 훈련 단계나 층을 만들었다면 디버깅이 필요할 것이다. 디버깅 수단은
프레임워크의 필수 요소다. 케라스에선 사용자를 고려하여 디버깅 작업 흐름이
설계돼 있다.

기본적으로 케라스 모델은 빠른 실행을 위해서 고도로 최적화된 계산 그래프로
컴파일되어 돈다. 즉, 여러분이 (가령 `train_step`에) 작성한 파이썬 코드가
실제 실행되는 코드가 아니다. 이런 간접 단계가 디버깅을 어렵게 만들 수 있다.

디버깅은 한 단계씩 진행하며 할 수 있으면 최고다. 그래서 코드 여기저기 `print()`
문을 집어넣어서 어떤 연산 후 데이터가 어떻게 되는지 보고 싶어 하는 것이고,
또 그래서 `pdb`를 쓰고 싶어 하는 것이다. **모델을 열심 방식으로 돌리면**
그럴 수 있다. 열심 실행 방식에선 작성한 파이썬 코드가 곧 실행되는 코드다.

`compile()`에 `run_eagerly=True`만 주면 된다.

```python
model.compile(optimizer='adam', loss='mse', run_eagerly=True)
```

단점은 당연히 모델이 상당히 느려진다는 점이다. 디버깅을 마쳤으면 꼭 원래대로
되돌려서 컴파일된 계산 그래프의 장점을 누리자.

일반적으로 `fit()` 호출 내에서 일어나는 뭔가를 디버깅해야 할 때마다
`run_eagerly=True`를 쓰게 될 것이다.

---
## 여러 GPU로 훈련 속도 높이기

케라스에는 `tf.distribute` API를 통해 업계 수준의 다중 GPU 훈련과
분산 다중 장비 훈련을 지원한다.

머신에 여러 GPU가 있다면 다음처럼 해서 모든 GPU에서 모델을 훈련시킬 수 있다.

- `tf.distribute.MirroredStrategy` 객체 생성
- 전략의 스코프 안에서 모델을 만들고 컴파일
- 평소처럼 데이터셋으로 `fit()` 및 `evaluate()` 호출

```python
# MirroredStrategy 만들기
strategy = tf.distribute.MirroredStrategy()

# 전략 스코프 열기
with strategy.scope():
  # 변수를 만드는 모든 동작이 전략 스코프 안에 있어야 한다.
  # 일반적으로 모델 구성과 `compile()`만 그에 해당한다.
  model = Model(...)
  model.compile(...)

# 모든 가용 장치에서 모델 훈련시키기
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=2, validation_data=val_dataset)

# 모든 가용 장치에서 모델 검사하기
model.evaluate(test_dataset)
```

다중 GPU 및 분산 훈련에 대한 자세한 소개는
[이 안내서](/guides/distributed_training/)를 보라.

---
## 전처리를 장치에서 동기적으로, 또는 호스트 CPU에서 비동기적으로 하기

앞서 전처리에 대해 배우면서 이미지 전처리 층들(`CenterCrop`과 `Rescaling`)을
모델 안에 직접 집어넣는 예시를 살펴봤다.

장치 상에서 전처리를 하고 싶다면, 예를 들어 GPU 가속으로 피처 정규화나 이미지
증강을 하고 싶다면 훈련 동안 모델의 일부로서 전처리가 이뤄지게 하는 게 최고다.
하지만 그런 구성에 적합하지 않은 종류의 전처리가 있다. 특히 `TextVectorization`
층을 이용한 텍스트 전처리가 그렇다. 순차적 동작 특성과 CPU에서만 돌 수 있다는
점 때문에 **비동기 전처리**가 나은 경우가 많다.

비동기 전처리 방식에선 전처리 연산들이 CPU에서 돌게 되고, GPU에서 앞선 데이터
배치를 처리하느라 바쁜 동안 전처리한 표본들이 큐에 버퍼링된다. 그리고 다시
GPU가 한가해지기 바로 전에 전처리된 다음 차례 표본 배치가 큐에서 GPU 메모리로
옮겨지게 된다. 전처리가 진행을 막지 않게 되므로 GPU를 최대한으로 돌릴 수 있다.

비동기 전처리를 하려면 `data.map`을 이용해 데이터 파이프라인에 전처리
동작을 끼워넣어 주기만 하면 된다.


```python
# dtype이 `string`인 예시 훈련 데이터
samples = np.array([["This is the 1st sample."], ["And here's the 2nd sample."]])
labels = [[0], [1]]

# TextVectorization 층 준비
vectorizer = TextVectorization(output_mode="int")
vectorizer.adapt(samples)

# 비동기 전처리: 텍스트 벡터화가 tf.data 파이프라인의 일부로서 이뤄진다.
# 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# 표본에 텍스트 벡터화 적용
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# 배치 2개 크기 버퍼로 미리 옮겨 두기
dataset = dataset.prefetch(2)

# 모델 입력으로 정수 열을 받아야 한다.
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 13ms/step - loss: 0.5028

<tensorflow.python.keras.callbacks.History at 0x147777490>

```
</div>
이를 모델의 일부로서 텍스트 벡터화를 하는 방식과 비교해 보자.


```python
# 데이터셋은 문자열 표본들을 내놓을 것이다.
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)

# 모델 입력으로 문자열을 받아야 한다.
inputs = keras.Input(shape=(1,), dtype="string")
x = vectorizer(inputs)
x = layers.Embedding(input_dim=10, output_dim=32)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 16ms/step - loss: 0.5258

<tensorflow.python.keras.callbacks.History at 0x1477b1910>

```
</div>
CPU에서 텍스트 모델을 훈련시킬 때는 일반적으로 두 구성의 성능 차이가 눈에 띄지
않을 것이다. 하지만 GPU에서 훈련시킬 때는 GPU에서 모델을 돌리는 동안 호스트
CPU에서 비동기 버퍼링 전처리를 하는 게 상당한 성능 향상을 가져올 수 있다.

훈련 후에 전처리 층(들)을 포함한 전구간 모델을 얻고 싶다면 쉽게 가능하다.
`TextVectorization`이 층이기 때문이다.

```python
inputs = keras.Input(shape=(1,), dtype='string')
x = vectorizer(inputs)
outputs = trained_model(x)
end_to_end_model = keras.Model(inputs, outputs)
```

---
## 하이퍼파라미터 조정으로 최적의 모델 구성 찾기

잘 동작하는 모델이 갖춰지고 나면 그 구성(여러 구조적 선택, 층 크기 등)을
최적화하고 싶을 것이다. 사람의 직감은 어느 정도까지만 통하기 때문에 이제는
체계적인 방법을 써야 하는데, 바로 하이퍼파라미터 탐색을 활용할 때다.

[KerasTuner](/api/keras_tuner/tuners/)를 이용해 케라스 모델에
최적인 하이퍼파라미터를 찾을 수 있다. `fit()` 호출만큼이나 간단하다.

이용 방식은 다음과 같다.

첫째로, `hp` 인자만 받는 함수를 만들어서 모델 정의를 그리 옮긴다. 그리고
조정하려는 값들을 `hp.Int()`나 `hp.Choice()` 같은 하이퍼파라미터 값 제공
메서드로 바꾼다.

```python
def build_model(hp):
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))(inputs)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
```

그 함수는 컴파일된 모델을 반환해야 한다.

다음으로, 최적화 목표와 여타 탐색 매개변수들을 지정해서 튜너 객체를 만든다.


```python
import keras_tuner

tuner = keras_tuner.tuners.Hyperband(
  build_model,
  objective='val_loss',
  max_epochs=100,
  max_trials=200,
  executions_per_trial=2,
  directory='my_dir')
```

마지막으로, `search()` 메서드로 탐색을 시작한다. `Model.fit()`과 같은 인자들을 받는다.

```python
tuner.search(dataset, validation_data=val_dataset)
```

탐색이 끝났으면 다음처럼 최적 모델(들)을 얻을 수 있다.

```python
models = tuner.get_best_models(num_models=2)
```

또는 결과 요약 정보를 찍을 수 있다.

```python
tuner.results_summary()
```

---
## 전구간 예시

이 소개서의 개념들에 익숙해지도록 다음과 같은 전구간 예시들을 살펴보자.

- [텍스트 분류](/examples/nlp/text_classification_from_scratch/)
- [이미지 분류](/examples/vision/image_classification_from_scratch/)
- [신용카드 부정사용 탐지](/examples/structured_data/imbalanced_classification/)

---
## 다음으로 배울 것들

- [함수형 API](/guides/functional_api/) 더 배우기.
- [`fit()`과 `evaluate()`의 기능](/guides/training_with_built_in_methods/) 더 배우기.
- [콜백](/guides/writing_your_own_callbacks/) 더 배우기.
- [자체 훈련 단계 만들기](/guides/customizing_what_happens_in_fit/) 더 배우기.
- [다중 GPU 훈련과 분산 훈련](/guides/distributed_training/) 더 배우기.
- [전이 학습](/guides/transfer_learning/) 하는 법 배우기.
