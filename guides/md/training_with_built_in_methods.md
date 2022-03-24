# 케라스 내장 메서드를 이용한 훈련과 평가

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2019/03/01<br>
**최근 변경:** 2020/04/13<br>
**설명:** `fit()`과 `evaluate()`를 이용한 훈련 및 평가에 대한 안내서.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/training_with_built_in_methods.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/training_with_built_in_methods.py)



---
## 준비


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---
## 소개

이 안내서는 훈련과 평가를 위한 (`Model.fit()`, `Model.evaluate()`,
`Model.predict()` 같은) 내장 API를 이용한 모델 훈련과 평가, 예측(추론)을
다룬다.

`fit()`을 활용하면서 자체적인 훈련 단계 함수를 쓰는 방식에 대해 알고 싶다면
[`fit()` 내부 동작 바꾸기 안내서](/guides/customizing_what_happens_in_fit/)를
보면 된다.

자체적인 훈련 루프와 평가 루프를 바닥부터 작성하는 방식에 대해 알고 싶다면
["훈련 루프 바닥부터 작성하기"](/guides/writing_a_training_loop_from_scratch/)
안내서를 보면 된다.

일반적으로 말해 내장 루프를 사용하건 자체 루프를 작성하건
모델 훈련과 평가 동작은 모든 케라스 모델에서 (순차형 모델,
함수형 API로 만든 모델, 서브클래스를 통해 바닥부터 작성한 모델 모두에서)
정확히 같은 방식으로 이뤄진다.

이 안내서에서 분산 훈련은 다루지 않는다.
[다중 GPU 훈련과 분산 훈련 안내서](/guides/distributed_training/)를 보면 된다.

---
## API 살펴보기: 첫 번째 전범위 예시

모델의 내장 훈련 루프에 데이터를 줄 때는 (데이터가 작아서 메모리에 들어가는
경우) **NumPy 배열**을 쓰거나 아니면 **`tf.data.Dataset` 객체**를 써야 한다.
이어지는 내용에선 최적화와 손실, 지표 사용 방식을 보이기 위해
NumPy 배열로 된 MNIST 데이터셋을 사용할 것이다.

다음 모델을 생각해 보자. (여기선 함수형 API로 모델을 만들지만
순차형 모델이나 서브클래스 모델도 가능하다.)


```python
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

그러면 전체 작업 흐름은 보통 다음 단계들로 이뤄진다.

- 훈련시키기
- 원래 훈련 데이터에서 따로 떼어 둔 일부를 가지고 평가하기
- 테스트 데이터를 가지고 검사하기

MNIST 데이터를 사용할 것이다.


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# (NumPy 배열로 된) 데이터 전처리하기
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# 10,000개 표본을 평가용으로 떼어 두기
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]
```

훈련 설정(최적화, 손실, 지표)을 지정하자.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # 최적화
    # 최소화할 손실 함수
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # 감시할 지표 목록
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

`fit()`을 호출하면 데이터를 `batch_size` 크기인 "배치"들로 나눠서 모델을 훈련시키며,
전체 데이터셋을 `epochs` 번만큼 반복해서 돈다.


```python
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # 에포크 끝마다 평가 손실과 지표들을
    # 감시할 수 있도록 평가용 데이터를
    # 좀 주자.
    validation_data=(x_val, y_val),
)
```

<div class="k-default-codeblock">
```
Fit model on training data
Epoch 1/2
782/782 [==============================] - 2s 2ms/step - loss: 0.3479 - sparse_categorical_accuracy: 0.9018 - val_loss: 0.2048 - val_sparse_categorical_accuracy: 0.9370
Epoch 2/2
782/782 [==============================] - 1s 2ms/step - loss: 0.1592 - sparse_categorical_accuracy: 0.9521 - val_loss: 0.1377 - val_sparse_categorical_accuracy: 0.9594

```
</div>
반환되는 `history` 객체는 훈련 중 손실 값과 지표 값의 변화 기록을 담고 있다.


```python
history.history
```




<div class="k-default-codeblock">
```
{'loss': [0.34790968894958496, 0.1592278927564621],
 'sparse_categorical_accuracy': [0.9017800092697144, 0.9521200060844421],
 'val_loss': [0.20476257801055908, 0.13772223889827728],
 'val_sparse_categorical_accuracy': [0.9369999766349792, 0.9593999981880188]}

```
</div>
`evaluate()`을 통해 테스트 데이터를 가지고 모델을 평가한다.


```python
# `evaluate`로 테스트 데이터에 대해 모델 평가하기
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# `predict`로 새 데이터에 대해
# 예측(확률들. 마지막 층의 출력) 만들어 내기
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
```

<div class="k-default-codeblock">
```
Evaluate on test data
79/79 [==============================] - 0s 1ms/step - loss: 0.1408 - sparse_categorical_accuracy: 0.9567
test loss, test acc: [0.14082984626293182, 0.9567000269889832]
Generate predictions for 3 samples
1/1 [==============================] - 0s 80ms/step
predictions shape: (3, 10)

```
</div>
이제 이 작업 흐름의 각 부분을 자세히 살펴보자.

---
## `compile()` 메서드: 손실, 지표, 최적화 지정하기

`fit()`으로 모델을 훈련시키려면 손실 함수와 최적화 방법을,
그리고 필요시 감시할 지표들을 지정해 줘야 한다.

그 값들을 `compile()` 메서드 인자로 모델에 준다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
```

`metrics` 인자는 리스트여야 한다. 모델에 지표가 여러 개일 수 있기 때문이다.

모델에 출력이 여러 개라면 각 출력마다 손실과 지표를 따로 지정할 수 있으며
각 출력이 모델 전체 손실에 기여하는 정도를 조정할 수 있다.
자세한 내용은 **입력과 출력이 여럿인 모델에 데이터 주기** 절을 보라.

참고로 기본 설정만으로 충분하다면 최적화 방식과 손실, 지표를
간단하게 문자열 식별자로 지정할 수도 있다.


```python
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
```

나중에 다시 이용할 수 있도록 모델 정의와 컴파일 단계를 함수로 만들어 두자.
이 안내서의 예시들에서 여러 번 호출하게 될 것이다.


```python

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

```

### 다양한 내장 최적화, 손실, 지표

왠만하면 손실이나 지표, 최적화 방식을 바닥부터 새로 만들 일이 없을 것이다.
필요한 게 케라스 API에 이미 포함돼 있을 것이기 때문이다.

최적화:

- `SGD()` (모멘텀 지정 가능)
- `RMSprop()`
- `Adam()`
- 등등

손실:

- `MeanSquaredError()`
- `KLDivergence()`
- `CosineSimilarity()`
- 등등

지표:

- `AUC()`
- `Precision()`
- `Recall()`
- 등등

### 자체 작성 손실

새로운 손실을 만들어야 한다면 두 가지 방법이 있다.

첫 번째 방식에선 입력 `y_true`와 `y_pred`를 받는 함수를 만들게 된다.
다음 예는 실제 데이터와 예측 간의 평균 제곱 오차를 계산하는 손실 함수를 보여 준다.


```python

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

# MSE 계산을 위해 레이블들을 원핫 인코딩
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.0162

<keras.callbacks.History at 0x159159fd0>

```
</div>
`y_true`와 `y_pred` 외의 매개변수도 받는 손실 함수가 필요하다면
`tf.keras.losses.Loss`의 서브클래스를 만들어서 다음 두 메서드를 구현하면 된다.

- `__init__(self)`: 손실 함수 호출에 전달할 매개변수들을 받는다.
- `call(self, y_true, y_pref)`: 목표치(y_true)와 모델 예측치(y_pred)를
사용해 모델의 손실을 계산한다.

평균 제곱 오차를 쓰되 (분류 목표치가 원핫 인코딩이어서 0에서 1 사이
값이라 하고) 예측치가 0.5에서 너무 멀지 않게 장려하는 항을 추가하고
싶다고 하자. 이렇게 하면 모델이 너무 확신하는 결과를 내지 않게 만들어서
과적합을 줄이는 데 도움이 된다. (진짜 그렇게 될까 싶다면 돌려 보자!)

다음처럼 된다.


```python

class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.0392

<keras.callbacks.History at 0x1599fd650>

```
</div>
### 자체 작성 지표

API에 포함 안 된 지표가 필요하다면 `tf.keras.metrics.Metric`의 서브클래스를
만들어서 손쉽게 자체 지표를 만들 수 있다. 4가지 메서드를 구현해야 한다.

- `__init__(self)`: 지표를 위한 상태 변수들을 만든다.
- `update_state(self, y_true, y_pred, sample_weight=None)`: 목표치 y_true와
모델 예측치 y_pred를 사용해 상태 변수를 갱신한다.
- `result(self)`: 상태 변수들을 가지고 최종 결과를 계산한다.
- `reset_state(self)`: 지표의 상태를 다시 초기화한다.

상태 갱신과 결과 계산이 (각기 `update_state()`와 `result()`로) 분리돼 있는
이유는 결과 계산 비용이 아주 커서 주기적으로만 수행하고 싶은 경우도 있기
때문이다.

다음 예시는 몇 개 표본이 올바로 분류됐는지를 세는 `CategoricalTruePositives`
지표 구현을 보여 준다.


```python

class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # 에포크가 시작될 때마다 지표 상태가 초기화된다
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)
```

<div class="k-default-codeblock">
```
Epoch 1/3
782/782 [==============================] - 2s 2ms/step - loss: 0.3414 - categorical_true_positives: 45121.0000
Epoch 2/3
782/782 [==============================] - 2s 2ms/step - loss: 0.1533 - categorical_true_positives: 47725.0000
Epoch 3/3
782/782 [==============================] - 1s 2ms/step - loss: 0.1120 - categorical_true_positives: 48333.0000

<keras.callbacks.History at 0x159b4b250>

```
</div>
### 표준 시그니처에 맞지 않는 손실과 지표 다루기

거의 대부분의 손실과 지표는 `y_true`와 모델 출력 `y_pred`를 가지고
계산할 수 있지만 다 그렇지는 않다. 예를 들어 정칙화 손실에는
층의 활성만 필요할 수 있는데 (이 경우 목표치가 없다), 그 활성이
모델 출력이 아닐 수도 있다.

그런 경우에는 새로운 층의 call 메서드에서 `self.add_loss(loss_value)`를
호출하면 된다. 이런 식으로 추가한 손실이 훈련 동안 (`compile()`에 줬던)
"주" 손실에 더해진다. 다음 예시에선 활성 정칙화를 추가한다. (참고로
활성 정칙화는 모든 케라스 층에 내장돼 있다. 이 층은 구체적 예시를
위한 것일 뿐이다.)


```python

class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # 그대로 통과


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# 활성 정칙화를 층 형태로 추가
x = ActivityRegularizationLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# 정칙화 요소 때문에 이전보다 훨씬 높은 손실이 표시된다.
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 2.4753

<keras.callbacks.History at 0x159cb87d0>

```
</div>
지표 값 기록에도 마찬가지로 `add_metric()`을 쓸 수 있다.


```python

class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # `aggregation` 인자는 각 에포크마다
        # 배치별 값들을 어떻게 종합할지를 정한다.
        # 여기선 그냥 평균한다.
        self.add_metric(
            keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
        )
        return inputs  # 그대로 통과


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# 표준 편차 기록을 층 형태로 삽입.
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.3363 - std_of_activation: 0.9996

<keras.callbacks.History at 0x159e1dbd0>

```
</div>
[함수형 API](/guides/functional_api/)에선
`model.add_loss(loss_tensor)`나
`model.add_metric(metric_tensor, name, aggregation)`을
호출할 수 있다.

간단한 예를 보자.


```python
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)

model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 2.5326 - std_of_activation: 0.0021

<keras.callbacks.History at 0x159f9e690>

```
</div>
한편으로 `add_loss()`를 통해 손실을 주면 손실 함수 없이 `compile()`을
호출하는 게 가능해진다. 최소화할 손실이 모델에 있기 때문이다.

다음 `LogisticEndpoint` 층을 살펴보자. 입력으로 목표치와 로짓을 받아서
`add_loss()`를 통해 교차 엔트로피 손실을 추적한다. 또한 `add_metric()`을
통해 분류 정확도를 추적한다.


```python

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # 훈련 시점 손실 값을 계산해서 `self.add_loss()`로
        # 층에 더한다.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # 정확도를 지표로 삼아서 `self.add_metric()`으로
        # 층에 더한다.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # (`.predict()`를 위해) 추론 시점 예측 텐서 반환.
        return tf.nn.softmax(logits)

```

이를 다음처럼 입력이 둘(입력 데이터와 목표치)이고 `loss` 인자 없이
컴파일하는 모델에 쓸 수 있다.


```python
import numpy as np

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")  # 손실 인자 없음!

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 214ms/step - loss: 0.8886 - binary_accuracy: 0.0000e+00

<keras.callbacks.History at 0x15a15fa90>

```
</div>
입력이 여럿인 모델을 훈련시키는 방법에 대한 자세한 내용은
**입력과 출력이 여럿인 모델에 데이터 주기** 절을 보라.

### 검사용 세트를 자동으로 떼어 두기

앞서 본 첫 번째 전범위 예시에선 `validation_data` 인자를 통해
NumPy 배열들의 튜플 `(x_val, y_val)`을 모델로 주었는데,
그걸 가지고 에포크 끝마다 검사 손실과 검사 지표들을 평가한다.

또 다른 방식이 있다. `validation_split` 인자를 쓰면 평가용 데이터 일부를
자동으로 검사용으로 떼어 둔다. 인자 값이 검사용으로 떼어 둘 데이터의
비율을 나타내므로 0보다 크고 1보다 작은 수로 설정하면 된다. 예를 들어
`validation_split=0.2`는 "데이터 중 20%를 검사에 사용하라"는 뜻이고
`validation_split=0.6`은 "데이터 중 60%를 검사에 사용하라"는 뜻이다.

`fit()` 호출에서 받은 배열들에서 어떤 뒤섞기도 하지 않은 상태로
뒤쪽 x% 표본들을 뽑아내는 방식으로 검사용 몫을 정한다.

NumPy 데이터로 훈련시킬 때만 `validation_split`을 쓸 수 있다.


```python
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)
```

<div class="k-default-codeblock">
```
625/625 [==============================] - 2s 2ms/step - loss: 0.3593 - sparse_categorical_accuracy: 0.8974 - val_loss: 0.2190 - val_sparse_categorical_accuracy: 0.9318

<keras.callbacks.History at 0x15a223bd0>

```
</div>
---
## tf.data Dataset으로 훈련시키고 평가하기

앞선 내용에서 손실과 지표, 최적화를 다루는 방법을 보았고 `fit()`에
NumPy 배열 데이터를 줄 때 `validation_data` 및 `validation_split` 인자를
쓰는 방법을 보았다.

이번엔 데이터가 `tf.data.Dataset` 객체 형태로 오는 경우를 살펴보자.

텐서플로 2.0에 있는 `tf.data` API는 데이터를 빠르고 확장성 있게 적재하고
전처리하기 위한 유틸리티 모음이다.

`Dataset`을 만드는 방법에 대한 자세한 설명은
[tf.data 문서](https://www.tensorflow.org/guide/data)를 보라.

`fit()`, `evaluate()`, `predict()` 메서드에 `Dataset` 인스턴스를 바로
줄 수 있다.


```python
model = get_compiled_model()

# 먼저 훈련용 Dataset 인스턴스를 만들자.
# 예시일 뿐이므로 앞서의 MNIST 데이터를 그대로 쓰자.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 데이터셋을 섞고 나눈다.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 이번엔 테스트용 데이터셋이다.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

# 데이터셋 자체에서 배치 처리를 해 주기 때문에
# `batch_size` 인자를 주지 않는다.
model.fit(train_dataset, epochs=3)

# 데이터셋에 대해 평가나 예측도 할 수 있다.
print("Evaluate")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))
```

<div class="k-default-codeblock">
```
Epoch 1/3
782/782 [==============================] - 2s 2ms/step - loss: 0.3358 - sparse_categorical_accuracy: 0.9046
Epoch 2/3
782/782 [==============================] - 2s 2ms/step - loss: 0.1540 - sparse_categorical_accuracy: 0.9544
Epoch 3/3
782/782 [==============================] - 1s 2ms/step - loss: 0.1109 - sparse_categorical_accuracy: 0.9663
Evaluate
157/157 [==============================] - 0s 1ms/step - loss: 0.1118 - sparse_categorical_accuracy: 0.9659

{'loss': 0.11180760711431503,
 'sparse_categorical_accuracy': 0.9659000039100647}

```
</div>
참고로 각 에포크가 끝날 때마다 Dataset이 재설정되기 때문에
다음 에포크에서 재사용할 수 있다.

Dataset의 몇 개 배치로만 훈련을 돌리고 싶다면 `steps_per_epoch` 인자를
줄 수 있다. 모델에서 그 Dataset으로 훈련 단계를 몇 번 돌고 나서 다음
에포크로 넘어가야 하는지를 지정한다.

이렇게 하면 각 에포크 끝에서 데이터셋이 재설정되지 않고 계속해서 다음 배치를
뽑아낸다. 그래서 (무한히 도는 데이터셋이 아닌 한) 결국 데이터가 다 떨어지게 된다.


```python
model = get_compiled_model()

# 훈련용 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 에포크마다 100개 배치만 (즉 64 * 100개 표본만) 사용
model.fit(train_dataset, epochs=3, steps_per_epoch=100)
```

<div class="k-default-codeblock">
```
Epoch 1/3
100/100 [==============================] - 1s 2ms/step - loss: 0.7515 - sparse_categorical_accuracy: 0.8031
Epoch 2/3
100/100 [==============================] - 0s 2ms/step - loss: 0.3731 - sparse_categorical_accuracy: 0.8919
Epoch 3/3
100/100 [==============================] - 0s 2ms/step - loss: 0.3165 - sparse_categorical_accuracy: 0.9084

<keras.callbacks.History at 0x15a405e90>

```
</div>
### 검사용 데이터셋으로 쓰기

`fit()`의 `validation_data` 인자로 `Dataset` 인스턴스를 줄 수 있다.


```python
model = get_compiled_model()

# 훈련용 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 평가용 데이터셋 준비
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(train_dataset, epochs=1, validation_data=val_dataset)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.3322 - sparse_categorical_accuracy: 0.9050 - val_loss: 0.1804 - val_sparse_categorical_accuracy: 0.9483

<keras.callbacks.History at 0x15a530510>

```
</div>
각 에포크가 끝날 때마다 모델에서 평가용 데이터셋을 돌려서 평가 손실과
평가 지표를 계산한다.

그 데이터셋의 몇 개 배치로만 평가를 돌리고 싶다면 `validation_steps`
인자를 줄 수 있다. 모델에서 평가용 데이터셋으로 평가 단계를 몇 번
돌고 나서 평가를 멈추고 다음 에포크로 넘어가야 하는지를 지정한다.


```python
model = get_compiled_model()

# 훈련용 데이터셋 준비
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 평가용 데이터셋 준비
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

model.fit(
    train_dataset,
    epochs=1,
    # `validation_steps` 인자를 써서 데이터셋의 처음 10개 배치만 가지고
    # 평가를 돌린다.
    validation_data=val_dataset,
    validation_steps=10,
)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.3429 - sparse_categorical_accuracy: 0.9038 - val_loss: 0.2760 - val_sparse_categorical_accuracy: 0.9312

<keras.callbacks.History at 0x1663870d0>

```
</div>
참고로 사용 후에는 항상 평가용 데이터셋이 재설정된다.
(따라서 모든 에포크에서 항상 같은 표본들로 평가를 하게 된다.)

`Dataset` 객체를 가지고 훈련할 때는 `validation_split` 인자(훈련용 데이터
일부 떼어 두기)를 지원하지 않는다. 그 동작을 위해선 데이터셋 표본에 인덱스로
접근할 수 있어야 하는데 `Dataset` API로는 일반적으로 불가능하기 때문이다.

---
## 지원하는 다른 입력 형식들

NumPy 배열과 Eager 텐서, 텐서플로 `Dataset`뿐 아니라 Pandas 데이터프레임,
심지어 데이터와 레이블 배치를 내놓는 파이썬 제너레이터를 가지고도
케라스 모델을 훈련시킬 수 있다.

그 중에서도 `keras.utils.Sequence` 클래스는 병렬 처리에 안전하고 표본 뒤섞기가
가능한 파이썬 데이터 제너레이터를 만들 수 있는 간단한 인터페이스를 제공해 준다.

일반적으로 다음을 권장한다.

- 데이터가 작아서 메모리에 들어간다면 NumPy 입력 데이터 사용.
- 데이터셋이 크고 분산 훈련을 해야 한다면 `Dataset` 객체 사용.
- 데이터셋이 크고 텐서플로에서는 불가능한 여러 처리를 따로 파이썬으로
해 줘야 한다면 (가령 데이터 적재나 전처리에 외부 라이브러리를 써야 한다면)
`Sequence` 객체 사용.


---
## `keras.utils.Sequence` 객체를 입력으로 쓰기

`keras.utils.Sequence`의 서브클래스를 만들어서 두 가지 중요한 특성을 가진
파이썬 제너레이터를 얻을 수 있다.

- 병렬 처리 때도 잘 동작한다.
- 표본들을 뒤섞을 수 있다. (예: `fit()`에 `shuffle=True`를 줬을 때)

`Sequence` 서브클래스에서 다음 두 메서드를 구현해야 한다.

- `__getitem__`
- `__len__`

`__getitem__` 메서드는 배치 하나를 반환해야 한다.
에포크마다 데이터셋을 바꾸고 싶다면 `on_epoch_end`도 구현할 수 있다.

간단한 예를 보자.

```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# 여기서 `filenames`는 이미지 경로 리스트이고
# `labels`는 연계된 레이블들이다.

class CIFAR10Sequence(Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
               for filename in batch_x]), np.array(batch_y)

sequence = CIFAR10Sequence(filenames, labels, batch_size)
model.fit(sequence, epochs=10)
```

---
## 표본 가중치와 분류 가중치 사용하기

기본 설정에선 데이터셋 내 빈도에 따라 표본의 가중치가 정해진다.
표본 빈도와 무관하게 데이터에 가중치를 줄 수 있는 방법이 두 가지 있다.

* 분류 가중치
* 표본 가중치

### 분류 가중치

`Model.fit()`의 `class_weight` 인자에 딕셔너리를 줘서 이 가중치를
설정할 수 있다. 그 딕셔너리가 분류 인덱스를 가중치로 매핑하고,
그러면 그 분류에 속한 표본들에 그 가중치가 쓰이게 된다.

이를 이용해 리샘플링 없이 분류 균형을 맞출 수도 있고
특정 분류를 더 중요하게 다루도록 모델을 훈련시킬 수도 있다.

예를 들어 데이터에서 분류 "0"이 분류 "1"의 절반 정도만 등장한다면
`Model.fit(..., class_weight={0: 1., 1: 0.5})`라고 할 수 있다.

다음 NumPy 예시에선 분류 가중치를 이용해 5번째 분류(MNIST 데이터셋에서
숫자 "5")를 정확히 분류하도록 중요도를 높인다.


```python
import numpy as np

class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    # "5" 분류 가중치를 "2"로 설정해서
    # 그 분류를 두 배 중요하게 다루기
    5: 2.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}

print("Fit with class weight")
model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
Fit with class weight
782/782 [==============================] - 2s 2ms/step - loss: 0.3759 - sparse_categorical_accuracy: 0.8994

<keras.callbacks.History at 0x1664ff2d0>

```
</div>
### 표본 가중치

정밀한 제어가 필요하다면, 또는 분류 모델을 만드는 게 아니라면
"표본 가중치"를 쓸 수 있다.

- NumPy 데이터로 훈련 시: `Model.fit()`에 `sample_weight`
  인자를 주면 된다.
- `tf.data`나 기타 이터레이터로 훈련 시: 튜플
  `(input_batch, label_batch, sample_weight_batch)`를 내놓으면 된다.

"표본 가중치" 배열은 총 손실 계산 시 배치의 각 표본에 가중치를
얼마나 줘야 하는지 나타내는 수 배열이다. 불균형한 분류 문제들에
흔히 쓰인다. (드문 분류에 가중치를 더 준다는 발상이다.)

가중치를 1 아니면 0으로 해서 손실 함수의 *마스크*로 쓸 수도 있다.
(특정 표본이 총 손실에 기여하는 부분을 완전히 버리는 것이다.)


```python
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

print("Fit with sample weight")
model = get_compiled_model()
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)
```

<div class="k-default-codeblock">
```
Fit with sample weight
782/782 [==============================] - 2s 2ms/step - loss: 0.3855 - sparse_categorical_accuracy: 0.8971

<keras.callbacks.History at 0x166650090>

```
</div>
다음은 대응하는 `Dataset` 예시다.


```python
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

# 표본 가중치를 포함한 Dataset 만들기
# (반환 튜플의 세 번째 항목)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))

# 데이터셋 뒤섞고 자르기
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=1)
```

<div class="k-default-codeblock">
```
782/782 [==============================] - 2s 2ms/step - loss: 0.3739 - sparse_categorical_accuracy: 0.9020

<keras.callbacks.History at 0x1667b0e10>

```
</div>
---
## 입력과 출력이 여럿인 모델에 데이터 주기

앞선 예시들에서 우리는 단일 입력(`(764,)` 형태 텐서)에 단일
출력(`(764,)` 형태 예측 텐서)인 모델을 살펴보았다. 하지만
입력이나 출력이 여럿인 모델은 어떻게 해야 할까?

`(32, 32, 3)` 형태(`(height, width, channels)`)인 이미지 입력과
`(None, 10)` 형태(`(timesteps, features)`)인 시계열 입력이 있는
모델을 생각해 보자. 모델에서 그 두 입력의 조합을 가지고 계산해서
두 가지 출력을 내놓게 되는데, (`(1,)` 형태인) "점수"와
(`(5,)` 형태인) 5항 분류에 대한 확률 분포다.


```python
image_input = keras.Input(shape=(32, 32, 3), name="img_input")
timeseries_input = keras.Input(shape=(None, 10), name="ts_input")

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name="score_output")(x)
class_output = layers.Dense(5, name="class_output")(x)

model = keras.Model(
    inputs=[image_input, timeseries_input], outputs=[score_output, class_output]
)
```

어떤 모델인지 쉽게 알 수 있도록 그림으로 그려 보자.
(그림에 적힌 형태는 표본별 형태가 아니라 배치 형태다.)


```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```




    
![png](/img/guides/training_with_built_in_methods/training_with_built_in_methods_64_0.png)
    



출력별로 다른 손실을 지정할 수 있다. 컴파일 때 손실 함수들을 리스트로 주면 된다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
)
```

모델에 손실 함수 하나면 주면 모든 출력에 같은 손실 함수가 적용된다.
(이번 경우에는 적합하지 않은 방식이다.)

지표도 마찬가지다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics=[
        [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        [keras.metrics.CategoricalAccuracy()],
    ],
)
```

출력 층들에 이름을 줬으므로 출력별 손실과 지표를 딕셔너리로 지정할 수도 있다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
)
```

출력이 두 개를 넘어가면 이름을 지정해서 딕셔너리를 쓰기를 권한다.

`loss_weights` 인자를 써서 출력별 손실마다 다른 가중치를 주는 것도
가능하다. (예를 들어 예시 모델의 "점수" 손실에 특혜를 주기 위해
분류 손실 2배만큼의 중요도를 줄 수 있을 것이다.)


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
    loss_weights={"score_output": 2.0, "class_output": 1.0},
)
```

어떤 출력을 예측에만 쓰고 훈련에는 쓰지 않을 거라면 그에 대한 손실을
계산하지 않을 수도 있다.


```python
# 손실 리스트 버전
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()],
)

# 손실 딕셔너리 버전
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"class_output": keras.losses.CategoricalCrossentropy()},
)
```

입력이 여럿이거나 출력이 여럿인 모델 `fit()`에 데이터를 주는 건
컴파일 때 손실 함수를 지정하는 것과 비슷하다. 즉, (손실 함수를 받은
출력과 1:1 매핑되는) **NumPy 배열들의 리스트**나 **출력 이름을
NumPy 배열로 매핑하는 딕셔너리들**을 줄 수 있다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
)

# 더미 NumPy 데이터 생성하기
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# 리스트로 fit
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)

# 또는 딕셔너리로 fit
model.fit(
    {"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},
    batch_size=32,
    epochs=1,
)
```

<div class="k-default-codeblock">
```
4/4 [==============================] - 1s 5ms/step - loss: 14.4474 - score_output_loss: 0.8739 - class_output_loss: 13.5735
4/4 [==============================] - 0s 6ms/step - loss: 12.3280 - score_output_loss: 0.6432 - class_output_loss: 11.6848

<keras.callbacks.History at 0x166bb7490>

```
</div>
다음은 `Dataset`을 쓰는 경우다. NumPy 배열 방식과 비슷하게 `Dataset`이
딕셔너리들의 튜플을 반환하게 해야 한다.


```python
train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"img_input": img_data, "ts_input": ts_data},
        {"score_output": score_targets, "class_output": class_targets},
    )
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model.fit(train_dataset, epochs=1)
```

<div class="k-default-codeblock">
```
2/2 [==============================] - 0s 8ms/step - loss: 10.9884 - score_output_loss: 0.5419 - class_output_loss: 10.4466

<keras.callbacks.History at 0x1669ce250>

```
</div>
---
## 콜백 이용하기

케라스의 콜백이란 훈련 중 여러 시점(에포크 시작, 배치 끝, 에포크 끝 등)에
호출되는 객체다. 콜백을 이용해 다음과 같은 동작들을 구현할 수 있다.

- (기본 에포크별 평가에 더해서) 훈련 과정 여러 지점에서 모델 평가하기
- 정기적으로, 또는 정해진 정확도 기준을 넘을 때 모델 상태 저장하기
- 훈련이 정체기에 접어든 것 같을 때 모델 학습률 바꾸기
- 훈련이 정체기에 접어든 것 같을 때 상단 층 미세 조정하기
- 훈련이 끝나거나 어떤 성능 기준치를 넘었을 때 이메일이나 메신저 알림 보내기
- 등등

`fit()` 호출 시 콜백 리스트를 줄 수 있다.


```python
model = get_compiled_model()

callbacks = [
    keras.callbacks.EarlyStopping(
        # `val_loss`가 더 개선되지 않으면 훈련 중단
        monitor="val_loss",
        # "더 개선되지 않음"을 "1e-2 이상 줄어들지 않음"으로 정의
        min_delta=1e-2,
        # "더 개선되지 않음"을 또한 "최소 2 에포크 동안"으로 정의
        patience=2,
        verbose=1,
    )
]
model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    callbacks=callbacks,
    validation_split=0.2,
)
```

<div class="k-default-codeblock">
```
Epoch 1/20
625/625 [==============================] - 2s 2ms/step - loss: 0.3692 - sparse_categorical_accuracy: 0.8946 - val_loss: 0.2295 - val_sparse_categorical_accuracy: 0.9287
Epoch 2/20
625/625 [==============================] - 1s 2ms/step - loss: 0.1683 - sparse_categorical_accuracy: 0.9498 - val_loss: 0.1777 - val_sparse_categorical_accuracy: 0.9473
Epoch 3/20
625/625 [==============================] - 1s 2ms/step - loss: 0.1225 - sparse_categorical_accuracy: 0.9633 - val_loss: 0.1517 - val_sparse_categorical_accuracy: 0.9546
Epoch 4/20
625/625 [==============================] - 1s 2ms/step - loss: 0.0968 - sparse_categorical_accuracy: 0.9701 - val_loss: 0.1403 - val_sparse_categorical_accuracy: 0.9597
Epoch 5/20
625/625 [==============================] - 1s 2ms/step - loss: 0.0811 - sparse_categorical_accuracy: 0.9754 - val_loss: 0.1394 - val_sparse_categorical_accuracy: 0.9579
Epoch 6/20
625/625 [==============================] - 1s 2ms/step - loss: 0.0674 - sparse_categorical_accuracy: 0.9802 - val_loss: 0.1564 - val_sparse_categorical_accuracy: 0.9574
Epoch 6: early stopping

<keras.callbacks.History at 0x166c3fe50>

```
</div>
### 이용 가능한 다양한 내장 콜백들

케라스에는 다음과 같은 다양한 콜백들이 기본으로 포함돼 있다.

- `ModelCheckpoint`: 주기적으로 모델 저장하기
- `EarlyStopping`: 평가 지표가 더는 개선되지 않을 때 훈련 중단하기
- `TensorBoard`: 주기적으로 모델 로그 기록해서
[텐서보드](https://www.tensorflow.org/tensorboard)로 시각화할 수 있게 하기
(자세한 내용은 "시각화" 절 참고)
- `CSVLogger`: 손실 및 지표 데이터를 CSV 파일로 찍기
- 등등

전체 목록은 [콜백 문서](/api/callbacks/)를 보라.

### 자체 콜백 작성하기

기반 클래스 `keras.callbacks.Callback`을 확장해서 자체 콜백을 만들 수 있다.
콜백에서 클래스 속성 `self.model`을 통해 연계 모델에 접근할 수 있다.

[자체 콜백 작성하기 안내서](/guides/writing_your_own_callbacks/)를 꼭 읽어 보자.

다음은 훈련 동안 배치별 손실 값 리스트를 저장하는 간단한 예시다.


```python

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

```

---
## 모델 체크포인트 저장하기

상당히 큰 데이터셋으로 모델을 훈련시킬 때는 반드시 모델을 자주
저장해야 한다.

가장 쉬운 방법이 `ModelCheckpoint` 콜백이다.


```python
model = get_compiled_model()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        # 모델을 저장할 경로.
        # 그 아래 두 매개변수는 `val_loss` 점수가
        # 개선됐을 때만 현재 모델을 덮어 쓰라는 뜻이다.
        # 모델 저장 이름에 현재 에포크가 들어간다.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # `val_loss`가 개선된 경우에만 모델 저장
        monitor="val_loss",
        verbose=1,
    )
]
model.fit(
    x_train, y_train, epochs=2, batch_size=64, callbacks=callbacks, validation_split=0.2
)
```

<div class="k-default-codeblock">
```
Epoch 1/2
617/625 [============================>.] - ETA: 0s - loss: 0.3668 - sparse_categorical_accuracy: 0.8954
Epoch 1: val_loss improved from inf to 0.22688, saving model to mymodel_1
INFO:tensorflow:Assets written to: mymodel_1/assets
625/625 [==============================] - 2s 3ms/step - loss: 0.3645 - sparse_categorical_accuracy: 0.8960 - val_loss: 0.2269 - val_sparse_categorical_accuracy: 0.9332
Epoch 2/2
622/625 [============================>.] - ETA: 0s - loss: 0.1748 - sparse_categorical_accuracy: 0.9480
Epoch 2: val_loss improved from 0.22688 to 0.17561, saving model to mymodel_2
INFO:tensorflow:Assets written to: mymodel_2/assets
625/625 [==============================] - 2s 2ms/step - loss: 0.1750 - sparse_categorical_accuracy: 0.9480 - val_loss: 0.1756 - val_sparse_categorical_accuracy: 0.9477

<keras.callbacks.History at 0x15a2f1910>

```
</div>
`ModelCheckpoint` 콜백을 이용해 장애 저항성을 구현할 수 있다.
의도치 않게 훈련이 중단된 경우에 마지막으로 저장된 모델 상태를 가지고
훈련을 다시 시작할 수 있다. 간단한 예를 보자.


```python
import os

# 체크포인트들을 저장할 디렉터리 준비
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # 마지막 모델을 복원한다. 저장된 체크포인트가 없으면
    # 새 모델을 생성한다.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
callbacks = [
    # 배치 100개마다 SavedModel을 저장한다.
    # 모델 저장 이름에 훈련 손실 값을 포함시킨다.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", save_freq=100
    )
]
model.fit(x_train, y_train, epochs=1, callbacks=callbacks)
```

<div class="k-default-codeblock">
```
Creating a new model
  67/1563 [>.............................] - ETA: 2s - loss: 1.1577 - sparse_categorical_accuracy: 0.6903INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.96/assets
 170/1563 [==>...........................] - ETA: 4s - loss: 0.7616 - sparse_categorical_accuracy: 0.7950INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.70/assets
 266/1563 [====>.........................] - ETA: 5s - loss: 0.6075 - sparse_categorical_accuracy: 0.8356INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.58/assets
 367/1563 [======>.......................] - ETA: 5s - loss: 0.5266 - sparse_categorical_accuracy: 0.8553INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.51/assets
 499/1563 [========>.....................] - ETA: 4s - loss: 0.4711 - sparse_categorical_accuracy: 0.8692INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.47/assets
 568/1563 [=========>....................] - ETA: 4s - loss: 0.4457 - sparse_categorical_accuracy: 0.8762INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.44/assets
 671/1563 [===========>..................] - ETA: 4s - loss: 0.4153 - sparse_categorical_accuracy: 0.8843INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.41/assets
 793/1563 [==============>...............] - ETA: 3s - loss: 0.3883 - sparse_categorical_accuracy: 0.8910INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.39/assets
 871/1563 [===============>..............] - ETA: 3s - loss: 0.3720 - sparse_categorical_accuracy: 0.8948INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.37/assets
 970/1563 [=================>............] - ETA: 2s - loss: 0.3554 - sparse_categorical_accuracy: 0.8993INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.35/assets
1095/1563 [====================>.........] - ETA: 2s - loss: 0.3369 - sparse_categorical_accuracy: 0.9045INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.34/assets
1199/1563 [======================>.......] - ETA: 1s - loss: 0.3227 - sparse_categorical_accuracy: 0.9080INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.32/assets
1297/1563 [=======================>......] - ETA: 1s - loss: 0.3138 - sparse_categorical_accuracy: 0.9102INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.31/assets
1395/1563 [=========================>....] - ETA: 0s - loss: 0.3073 - sparse_categorical_accuracy: 0.9121INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.31/assets
1473/1563 [===========================>..] - ETA: 0s - loss: 0.3010 - sparse_categorical_accuracy: 0.9140INFO:tensorflow:Assets written to: ./ckpt/ckpt-loss=0.30/assets
1563/1563 [==============================] - 8s 5ms/step - loss: 0.2943 - sparse_categorical_accuracy: 0.9159

<keras.callbacks.History at 0x167035e50>

```
</div>
모델을 저장하고 복원하는 새로운 콜백을 작성할 수도 있다.

직렬화와 저장에 대한 자세한 내용은
[모델 직렬화와 저장 안내서](/guides/serialization_and_saving/)를 보라.

---
## 학습률 스케줄 이용하기

딥 러닝 모델 훈련에 흔한 패턴 하나는 훈련이 진행됨에 따라 학습률을
점진적으로 낮추는 것이다. 이를 보통 "학습률 감쇄(decay)"라 한다.

학습률 감쇄는 (현재 에포크나 현재 배치 번호에 따라 미리 정해지는)
정적 방식일 수도 있고 (모델의 현재 동작, 특히 평가 손실에 따라 바뀌는)
동적 방식일 수도 있다.

### 최적화 객체에 감쇄 방식 주기

정적 학습률 감쇄 방식을 쓰려면 최적화 객체의 `learning_rate` 인자로
스케줄 객체를 주기만 하면 된다.


```python
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
```

`ExponentialDecay`, `PiecewiseConstantDecay`, `PolynomialDecay`, `InverseTimeDecay`
등의 다양한 스케줄을 이용할 수 있다.

### 콜백 이용해 동적 학습률 감쇄 구현하기

스케줄 객체를 가지고는 동적 학습률 감쇄가 (예를 들어 평가 손실 값이 더
개선되지 않을 때 학습률을 낮추는 게) 불가능하다. 최적화 객체에서 평가 지표에
접근할 수 없기 때문이다.

하지만 콜백에선 평가 지표를 포함해 모든 지표들에 접근이 가능하다. 따라서
최적화 객체의 현재 학습률을 바꾸는 콜백을 이용하면 동적 감쇄를 구현할 수 있다.
실제로 그 동작이 `ReduceLROnPlateau` 콜백으로 내장돼 있기도 하다.

---
## 훈련 중 손실과 지표 시각화하기

훈련 중 모델을 관찰하기에 가장 좋은 방법은
[텐서보드](https://www.tensorflow.org/tensorboard)를 이용하는 것이다.
로컬에서 돌릴 수 있는 브라우저 기반 응용이며 다음을 제공한다.

- 훈련과 평가의 손실 및 지표들의 실시간 그래프
- (선택적) 층 활성 히스토그램 시각화
- (선택적) `Embedding` 층에서 학습한 내장 공간 3차원 시각화

텐서플로를 pip로 설치했다면 다음 명령으로 텐서보드를 띄울 수 있을 것이다.

```
tensorboard --logdir=/full_path_to_your_logs
```

### 텐서보드 콜백 이용하기

케라스 모델과 `fit()` 메서드에서 텐서보드를 이용하는 가장 쉬운 방식은
`TensorBoard` 콜백이다.

가장 간단하게는 콜백에서 로그를 기록할 곳을 지정하는 것만으로 충분하다.


```python
keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_logs",
    histogram_freq=0,  # 히스토그램 시각화 로그 기록 빈도
    embeddings_freq=0,  # 내장 시각화 로그 기록 빈도
    update_freq="epoch",  # 로그 기록 빈도 (기본: 에포크당 한 번)
)
```




<div class="k-default-codeblock">
```
<keras.callbacks.TensorBoard at 0x12fa767d0>

```
</div>
더 자세한 내용은 [`TensorBoard` 콜백 문서](/api/callbacks/tensorboard/)를 보라.
