# 순차형 모델

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2020/04/12<br>
**최근 변경:** 2020/04/12<br>
**설명:** 순차형 모델에 대한 안내서.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/sequential_model.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/sequential_model.py)



---
## 준비


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---
## 순차형 모델을 써야 할 때

`Sequential` 모델은 **입력 텐서와 출력 텐서가 딱 1개씩인**
**층들을 평범하게 순차 연결**한 경우에 적합하다.

개략적으로 다음 `Sequential` 모델이


```python
# 층 3개짜리 순차형 모델 정의
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# 테스트 입력으로 모델 호출
x = tf.ones((3, 3))
y = model(x)
```

다음 함수와 동등하다.


```python
# 층 3개 만들기
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# 텍스트 입력으로 층들 호출
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
```

다음 경우에는 순차형 모델이 **적합하지 않다**.

- 모델 입력이나 출력이 여러 개다.
- 어느 층에서든 입력이나 출력이 여러 개다.
- 층 공유를 해야 한다.
- 비선형 구조(예: 잔여 연결, 다분기 모델)가 필요하다.

---
## 순차형 모델 만들기

Sequential 생성자에 층들의 목록을 줘서 순차형 모델을 만들 수 있다.


```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)
```

`layers` 속성을 통해 그 층들에 접근할 수 있다.


```python
model.layers
```




<div class="k-default-codeblock">
```
[<tensorflow.python.keras.layers.core.Dense at 0x7fbd5f285a00>,
 <tensorflow.python.keras.layers.core.Dense at 0x7fbd5f285c70>,
 <tensorflow.python.keras.layers.core.Dense at 0x7fbd5f285ee0>]

```
</div>
`add()` 메서드를 통해 점진적으로 순차형 모델을 만들 수도 있다.


```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

그에 대응하는 `pop()` 메서드로 층을 제거할 수도 있다.
순차형 모델은 층들의 리스트와 아주 비슷하게 동작한다.


```python
model.pop()
print(len(model.layers))  # 2
```

<div class="k-default-codeblock">
```
2

```
</div>
또한 케라스의 여느 층이나 모델과 마찬가지로 Sequential 생성자도
`name` 인자를 받는다. 이를 이용해 텐서보드 그래프에 의미 있는 이름을
표시할 수 있다.


```python
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```

---
## 입력 형태 미리 지정하기

일반적으로 케라스의 모든 층들은 입력의 형태를 알아야 가중치를
만들 수 있다. 따라서 다음처럼 층을 만들었을 때 처음에는 가중치가 없다.


```python
layer = layers.Dense(3)
layer.weights  # 비어 있음
```




<div class="k-default-codeblock">
```
[]

```
</div>
입력 형태에 따라 가중치의 형태가 정해지므로 입력을 가지고
층을 처음 호출할 때 가중치가 만들어진다.


```python
# 테스트 입력으로 층 호출하기
x = tf.ones((1, 4))
y = layer(x)
layer.weights  # (4, 3) 및 (3,) 형태의 가중치가 생겼다
```




<div class="k-default-codeblock">
```
[<tf.Variable 'dense_6/kernel:0' shape=(4, 3) dtype=float32, numpy=
 array([[-0.5312456 , -0.02559239, -0.77284306],
        [-0.18156391,  0.7774476 , -0.05044252],
        [-0.3559971 ,  0.43751895,  0.3434813 ],
        [-0.25133908,  0.8889308 , -0.6510118 ]], dtype=float32)>,
 <tf.Variable 'dense_6/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]

```
</div>
당연히 순차형 모델에서도 그렇다. 입력 형태 없이 순차형 모델 인스턴스를
만들면 모델이 "구축"되지 않는다. 즉, 가중치가 없다.
(그래서 `model.weights`를 호출하면 오류가 난다.)
모델에 어떤 입력 데이터가 처음 들어갈 때 가중치들이 만들어진다.


```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # 이 단계에선 가중치가 없다

# 이때 다음이 불가능하다
# model.weights

# 다음도 불가능하다
# model.summary()

# 테스트 입력으로 모델 호출하기
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6
```

<div class="k-default-codeblock">
```
Number of weights after calling the model: 6

```
</div>
모델이 "구축"되고 나면 `summary()` 메서드를 호출해서
모델 내용을 표시할 수 있다.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_7 (Dense)              (1, 2)                    10        
_________________________________________________________________
dense_8 (Dense)              (1, 3)                    9         
_________________________________________________________________
dense_9 (Dense)              (1, 4)                    16        
=================================================================
Total params: 35
Trainable params: 35
Non-trainable params: 0
_________________________________________________________________

```
</div>
하지만 순차형 모델을 점진적으로 구축할 때 지금까지 만든 모델의 요약 정보를
현재 출력 형태까지 포함해서 표시할 수 있다면 굉장히 편리한 경우가 있다.
그럴 때는 모델 구축을 시작할 때 모델에 `Input` 객체를 줄 수 있다.
그러면 처음부터 입력 형태를 알게 된다.


```python
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 2)                 10        
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

```
</div>
보다시피 `Input` 객체는 `model.layers` 표시 내용에 포함되지 않는다.
층이 아니기 때문이다.


```python
model.layers
```




<div class="k-default-codeblock">
```
[<tensorflow.python.keras.layers.core.Dense at 0x7fbd5f1776d0>]

```
</div>
또 다른 방법은 첫 번째 층에 `input_shape` 인자를 주는 것이다.


```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()
```

<div class="k-default-codeblock">
```
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_11 (Dense)             (None, 2)                 10        
=================================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________________________________

```
</div>
이렇게 미리 정해진 입력 형태로 구축한 모델들은 항상 (데이터를 받기 전부터)
가중치를 가지고 있으며 항상 출력 형태가 정해져 있다.

일반적으로 입력 형태를 알고 있다면 순차형 모델에서 항상 입력 형태를
미리 지정해 주기를 권장한다.

---
## 일반적인 디버깅 흐름: `add()` + `summary()`

순차형 구조를 새로 구축할 때 `add()`를 써서 점진적으로 층을 쌓아가면서
자주 모델 요약 정보를 찍어 보는 게 도움이 된다. 예를 들어 다음처럼
`Conv2D` 및 `MaxPooling2D` 층들이 이미지 피처 맵을 얼마나 다운샘플링
하는지 확인해 볼 수 있다.


```python
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB 이미지
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# 현재 출력 형태가 뭔지 알 수 있겠는가? 어려울 것이다.
# 찍어 보자.
model.summary()

# 답은 (40, 40, 32), 따라서 다운샘플링을 더 할 수 있다....

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# 지금은?
model.summary()

# 이제 4x4 피처 맵을 얻었으니 전역 최대 풀링을 적용할 차례다.
model.add(layers.GlobalMaxPooling2D())

# 마지막으로 분류 층을 추가한다.
model.add(layers.Dense(10))
```

<div class="k-default-codeblock">
```
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 123, 123, 32)      2432      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 121, 121, 32)      9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 40, 40, 32)        0         
=================================================================
Total params: 11,680
Trainable params: 11,680
Non-trainable params: 0
_________________________________________________________________
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 123, 123, 32)      2432      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 121, 121, 32)      9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 40, 40, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 38, 38, 32)        9248      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 36, 36, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 32)        9248      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 32)          9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 32)          0         
=================================================================
Total params: 48,672
Trainable params: 48,672
Non-trainable params: 0
_________________________________________________________________

```
</div>
아주 유용하다. 안 그런가?


---
## 모델을 만들고 나서 할 수 있는 일

모델 구조가 준비되고 나면 다음을 해 볼 수 있다.

- 모델 훈련시키고, 평가하고, 추론 돌리기.
[내장 루프를 이용한 훈련 및 평가 안내서](
    /guides/training_with_built_in_methods/) 참고
- 모델을 디스크에 저장하고 복원하기.
[직렬화와 저장 안내서](/guides/serialization_and_saving/) 참고.
- 여러 GPU 활용해서 모델 훈련 속도 높이기.
[다중 GPU 훈련과 분산 훈련 안내서](/guides/distributed_training/) 참고.

---
## 순차형 모델과 피처 추출

일단 순차형 모델을 구축하고 나면 [함수형 API 모델](/guides/functional_api/)과
비슷하게 동작한다. 그래서 층마다 `input` 속성과 `output` 속성이 있는데,
이 속성들을 이용해 재밌는 걸 할 수 있다. 예를 들어 순차 모델
중간 층들의 출력을 모두 추출하는 모델을 금방 만들 수 있다.


```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# 테스트 입력으로 피처 추출기 호출하기
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

다음은 한 층에서만 피처들을 추출하는 비슷한 예시다.


```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# 테스트 입력으로 피처 추출기 호출하기
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

---
## 순차형 모델과 전이 학습

전이 학습의 요체는 모델 하위 층들을 고정시키고 상위 층들만 훈련시키는 것이다.
익숙치 않다면 [전이 학습 안내서](/guides/transfer_learning/)를
꼭 읽어 보자.

순차 모델을 수반하는 흔한 전이 학습 방식 두 가지를 간략히 살펴보자.

첫 번째로, 어떤 순차 모델이 있고 마지막 층을 뺀 모든 층을 고정시키고
싶다고 해 보자. 이 경우는 `model.layers`를 순회하면서 마지막 층을 뺀
모든 층에서 `layer.trainable = False` 설정을 해 주기만 하면 된다.

```python
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# 먼저 사전 훈련된 가중치들을 적재해야 할 것이다.
model.load_weights(...)

# 마지막 층을 뺀 모든 층들을 고정시키기
for layer in model.layers[:-1]:
  layer.trainable = False

# 다시 컴파일해서 훈련 (마지막 층의 가중치만 갱신하게 된다.)
model.compile(...)
model.fit(...)
```

또 다른 방식은 순차 모델을 사용해서 사전 훈련 모델 위에
새로 초기화한 어떤 분류 층을 쌓는 것이다.

```python
# 합성곱 기반 모델에 사전 훈련된 가중치 적재하기
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# 기반 모델 고정시키기
base_model.trainable = False

# 순차 모델 사용해서 그 위에 훈련 가능한 분류 층 추가하기
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# 컴파일 및 훈련
model.compile(...)
model.fit(...)
```

전이 학습을 한다면 아마 이 두 패턴들을 자주 쓰게 될 것이다.

순차 모델에 대해 알아야 할 건 여기까지다!

케라스에서 모델을 구축하는 방법에 대해 더 알고 싶으면 다음을 보라.

- [함수형 API 안내서](/guides/functional_api/)
- [서브클래스로 새 층과 모델 만들기 안내서](
    /guides/making_new_layers_and_models_via_subclassing/)
