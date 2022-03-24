# 서브클래스로 새 층과 모델 만들기

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2019/03/01<br>
**최근 변경:** 2020/04/13<br>
**설명:** `Layer` 및 `Model` 객체를 바닥부터 작성하는 방법에 대한 안내서.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/making_new_layers_and_models_via_subclassing.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/making_new_layers_and_models_via_subclassing.py)



---
## 준비


```python
import tensorflow as tf
from tensorflow import keras
```

---
## `Layer` 클래스: 상태(가중치)와 연산

케라스의 핵심 개념 중 하나가 `Layer` 층이다. 상태(층의 "가중치")와
입력에서 출력으로의 변환("호출", 층 진행)을 캡슐화한 것이다.

다음은 밀집 연결 층이다. 변수 `w`와 `b`가 상태다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

```

파이썬 함수처럼 어떤 텐서 입력(들)을 가지고 호출하는 방식으로 층을
사용하게 된다.


```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[ 0.01103698  0.03099662 -0.1009444   0.10721317]
 [ 0.01103698  0.03099662 -0.1009444   0.10721317]], shape=(2, 4), dtype=float32)

```
</div>
가중치 `w`와 `b`를 층 속성으로 설정하고 나면 층에서 자동으로
그 가중치들을 추적한다.


```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

층에 가중치를 추가하는 더 빠른 방법도 있다. `add_weight()` 메서드를 쓰는 것이다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[-0.09724902  0.04435382  0.06548684  0.1264643 ]
 [-0.09724902  0.04435382  0.06548684  0.1264643 ]], shape=(2, 4), dtype=float32)

```
</div>
---
## 층에 훈련 불가능 가중치 두기

훈련 가능한 가중치뿐 아니라 훈련 불가능 가중치도 층에 추가할 수 있다.
훈련 불가능이란 건 층을 훈련시킬 때 역전파에 그 가중치를 포함시키지
않는다는 의미다.

다음처럼 훈련 불가능 가중치를 추가하고 사용할 수 있다.


```python

class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
```

<div class="k-default-codeblock">
```
[2. 2.]
[4. 4.]

```
</div>
`layer.weights`에 포함되지만 훈련 불가능 가중치로 분류돼 있다.


```python
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

# 훈련 가능 가중치 목록에 포함되지 않는다
print("trainable_weights:", my_sum.trainable_weights)
```

<div class="k-default-codeblock">
```
weights: 1
non-trainable weights: 1
trainable_weights: []

```
</div>
---
## 모범 설계: 입력 형태를 알게 될 때까지 가중치 생성 연기하기

앞서 만든 `Linear` 층은 `__init__()`에 `input_dim` 인자가 있어서 
가중치 `w`와 `b` 형태를 계산하는 데 사용한다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

```

하지만 많은 경우 입력 크기를 미리 알지 못할 수 있고, 그래서 층 인스턴스
생성 후 크기를 알게 됐을 때 가중치들을 만들고 싶을 수 있다.

케라스 API에선 다음처럼 층의 `build(self, inputs_shape)` 메서드에서
층 가중치들을 만들기를 권장한다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

```

층의 `__call__()` 메서드가 처음 호출될 때 자동으로 build가 실행된다.
이제 필요할 때가 돼야 동작을 수행하는, 그래서 쓰기 쉬운 층이 되었다.


```python
# 인스턴스 생성 시점엔 어떤 입력으로 층을 호출할지 알 수 없다.
linear_layer = Linear(32)

# 층이 처음 호출될 때 동적으로 층의 가중치들이 생성된다.
y = linear_layer(x)

```

위와 같이 `build()`를 분리해서 구현하면 가중치를 한 번 만드는 동작과
호출 여러 번에서 사용하는 동작이 깔끔하게 분리된다. 하지만 일부 특수한
자체 제작 층에선 상태 생성과 계산을 분리하는 게 비현실적일 수 있다.
그런 경우에는 가중치 생성 동작이 첫 번째 `__call__()`에서 이뤄지도록
지연시킬 수도 있지만 이후 호출들에서도 같은 가중치가 쓰이도록
신경 써 줘야 한다. 한편으로 `__call__()`이 처음 실행될 때는
`tf.function` 내부에서일 가능성이 높기 때문에 `__call__()`에서
이뤄지는 변수 생성을 `tf.init_scope`로 감싸 주어야 한다.

---
## 재귀적으로 층 만들기

어떤 Layer 인스턴스를 다른 Layer의 속성으로 할당하면
내부 층에서 만드는 가중치들을 바깥 층에서 추적하기 시작한다.

`__init__()`에서 그런 하위 층을 만들기를 권장하며, 가중치 생성을
촉발시키는 건 첫 번째 `__call__()`에게 맡겨 두자.


```python

class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # `mlp` 첫 호출 때 가중치들을 만든다
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
```

<div class="k-default-codeblock">
```
weights: 6
trainable weights: 6

```
</div>
---
## `add_loss()` 메서드

층의 `call()` 메서드를 작성할 때 이후 훈련 루프 작성 시 사용할
손실 텐서를 만들 수 있다. `self.add_loss(value)`를 호출하면 된다.


```python
# 활성 정칙화 손실 만드는 층
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

```

`layer.losses`를 통해 이 손실을 (내부 층에서 만든 손실 포함)
얻을 수 있다. 최상위 층 `__call__()` 시작 때마다 그 속성이
재설정되며, 그래서 `layer.losses`는 항상 지난번 진행 동안
만들어진 손실 값을 담고 있다.


```python

class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # 층이 호출되지 않았으므로 아직 손실 없음

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # 손실 값 하나 생성했음

# __call__ 시작 때마다 `layer.losses` 재설정
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # 위 호출 동안 만들어진 손실임
```

또한 `layer.losses` 속성은 내부 층 가중치에 대해 만들어진
정칙화 손실도 담는다.


```python

class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(tf.zeros((1, 1)))

# 위의 `kernel_regularizer`에 의해 만들어진 손실
# `1e-3 * sum(layer.dense.kernel ** 2)`
print(layer.losses)
```

<div class="k-default-codeblock">
```
[<tf.Tensor: shape=(), dtype=float32, numpy=0.0023243506>]

```
</div>
다음처럼 훈련 루프를 작성할 때 그 손실들을 가져다 쓰게 된다.

```python
# 최적화 인스턴스 만들기
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 데이터셋 배치들을 가지고 돌기
for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
    logits = layer(x_batch_train)  # 이 미니배치에 대한 로짓
    # 이 미니배치에 대한 손실 값
    loss_value = loss_fn(y_batch_train, logits)
    # 이 진행 동안 생긴 추가 손실 더하기
    loss_value += sum(model.losses)

  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

훈련 루프 작성에 대한 자세한 안내는
[훈련 루프 바닥부터 작성하기 안내서](/guides/writing_a_training_loop_from_scratch/)를 보라.

`fit()`을 쓸 때도 그 손실들이 매끄럽게 동작한다.
(즉, 자동으로 그 합이 주 손실에 더해진다.)


```python
import numpy as np

inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# `compile`에 손실을 주면 정칙화 손실이 거기 더해진다.
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# `compile`에 손실을 주지 않을 수도 있다. 진행 동안 `add_loss`
# 호출을 통해서 최소화할 손실이 모델에 생겼기 때문이다.
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 131ms/step - loss: 0.1269
1/1 [==============================] - 0s 45ms/step - loss: 0.0274

<keras.callbacks.History at 0x1643af310>

```
</div>
---
## `add_metric()` 메서드

`add_loss()`와 비슷하게 층에는 훈련 동안 측정치 이동 평균을
추적하기 위한 `add_metric()` 메서드가 있다.

"로지스틱 종점"이란 층을 생각해 보자. 입력으로 예측치와 목표치를
받고, 손실을 계산해서 `add_loss()`를 통해 추적하고, 정확도 스칼라를
계산해서 `add_metric()`을 통해 추적한다.


```python

class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # 훈련 시 손실 값 계산해서 `self.add_loss()`로 추가
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # 정확도를 지표로 얻어서 `self.add_metric()`으로 추가
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # 추론 시 (`.predict()`) 예측 텐서 반환
        return tf.nn.softmax(logits)

```

이런 식으로 추적하는 지표들을 `layer.metrics`를 통해 접근할 수 있다.


```python
layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print("layer.metrics:", layer.metrics)
print("current accuracy value:", float(layer.metrics[0].result()))
```

<div class="k-default-codeblock">
```
layer.metrics: [<keras.metrics.BinaryAccuracy object at 0x161505450>]
current accuracy value: 1.0

```
</div>
`add_loss()`와 마찬가지로 `fit()`에서 이 지표들도 추적한다.


```python
inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
```

<div class="k-default-codeblock">
```
1/1 [==============================] - 0s 240ms/step - loss: 0.9455 - binary_accuracy: 0.0000e+00

<keras.callbacks.History at 0x1644acd50>

```
</div>
---
## 필요시 층 직렬화 가능하게 하기

새로 만든 층이 [함수형 모델](/guides/functional_api/)에 포함돼 있어서
직렬화가 가능해야 한다면 `get_config()` 메서드를 구현하면 된다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


# 이제 설정을 가지고 층을 다시 만들 수 있다
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

<div class="k-default-codeblock">
```
{'units': 64}

```
</div>
한편 기반 클래스 `Layer`의 `__init__()` 메서드는 `name`과 `dtype` 같은
키워드 인자를 받는다. `__init__()`에서 그런 인자들을 부모 클래스로 전달하고
층 설정에도 포함시키는 게 좋다.


```python

class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

<div class="k-default-codeblock">
```
{'name': 'linear_8', 'trainable': True, 'dtype': 'float32', 'units': 64}

```
</div>
설정을 가지고 층을 역직렬화할 때 더 유연한 동작이 필요하다면
클래스 메서드 `from_config()`를 오버라이드하면 된다.
다음은 `from_config()` 기본 구현이다.

```python
def from_config(cls, config):
  return cls(**config)
```

직렬화와 저장에 대해 더 알고 싶다면
[모델 저장 및 직렬화 안내서](/guides/serialization_and_saving/)를 보라.

---
## `call()` 메서드의 특수 인자 `training`

`BatchNormalization` 층과 `Dropout` 층 같은 일부 층들은 훈련 때와 추론 때의
동작이 다르다. 그런 층들에선 `call()` 메서드에 `training`이라는 (불리언)
인자를 제공하는 게 표준 관행이다.

`call()`의 그 인자에 따라 내장 훈련 루프와 평가 루프(예: `fit()`)를 켜서
층을 훈련이나 추론으로 올바르게 돌릴 수 있다.


```python

class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

```

---
## `call()` 메서드의 특수 인자 `mask`

`call()`에서 지원하는 또 다른 특수 인자로 `mask`가 있다.

케라스의 RNN 층들에서 이 인자를 볼 수 있다. 마스크는 (입력의 단계마다
불리언 값이 하나씩 있는) 불리언 텐서다. 시계열 데이터 처리 시
특정 입력 단계들을 건너뛰는 데 쓴다.

앞선 층에서 마스크를 생성했으면 케라스에서 자동으로 마스크 지원 층의
`__call__()`로 올바른 `mask` 인자를 전달한다. 마스크 생성 층으로는
`mask_zero=True`로 설정한 `Embedding` 층, 그리고 `Masking` 층이 있다.

마스크 사용법과 마스킹 지원 층을 작성하는 방법에 대해 더 알고 싶다면
["마스킹과 패딩 이해하기"](/guides/understanding_masking_and_padding/)
안내서를 확인해 보자.

---
## `Model` 클래스

일반적으로 `Layer` 클래스를 써서 내부 연산 블록을 정의하고
`Model` 클래스를 써서 외부 모델을 정의하게 된다.
그 모델이 훈련 대상이 된다.

예를 들어 ResNet50 모델에선 `Layer` 서브클래스로 여러 ResNet 블록을 만들고
`Model` 하나로 전체 ResNet50 망을 포괄하게 한다.

`Model` 클래스는 `Layer`와 같은 API를 제공하면서도 다음 차이점이 있다.

- 내장된 훈련 루프, 평가 루프, 예측 루프를 제공한다.
(`model.fit()`, `model.evaluate()`, `model.predict()`)
- `model.layers` 속성을 통해 내부 층들의 목록을 제공한다.
- 저장 및 직렬화 API를 제공한다. (`save()`, `save_weights()`, ...)

실질적으로 `Layer` 클래스는 우리가 문헌에서 ("합성곱 층"이나
"순환 층"이라고 할 때의) "층"이나 ("ResNet 블록"이나
"Inception 블록"이라고 할 때의) "블록"에 대응한다.

반면 `Model` 클래스는 문헌에서 ("딥 러닝 모델"이라고 할 때의)
"모델"이나 ("심층 신경망"이라고 할 때의) "망"에 대응한다.

따라서 "`Layer` 클래스와 `Model` 클래스 중 어느 쪽을 써야 할까?"라는
생각이 든다면 이렇게 자문해 보면 된다. 그 대상에 `fit()`을 호출해야 할까?
그 대상에 `save()`를 호출해야 할까? 만약 그렇다면 `Model`로 가면 된다.
반면 (만들려는 클래스가 더 큰 시스템의 구성 블록일 뿐이어서,
또는 훈련 및 저장 코드를 직접 작성할 거라서) 그렇지 않다면
`Layer`를 쓰면 된다.

예를 들어 앞서의 미니 ResNet 예시를 가져다가
`fit()`으로 훈련시키고 `save_weights()`로 저장할 수 있는
`Model`을 만들어 보자.

```python
class ResNet(tf.keras.Model):

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save(filepath)
```

---
## 모두 합치기: 전범위 예시

지금까지 다음을 배웠다.

- `Layer`는 (`__init__()`이나 `build()`에서 만드는) 상태와
(`call()`에서 정의하는) 연산을 담는다.
- 층을 다른 층으로 감싸서 더 큰 연산 블록을 새로 만들 수 있다.
- 층에서 `add_loss()` 및 `add_metric()`을 통해 손실(보통 정칙화 손실)과
지표를 만들고 추적할 수 있다.
- 훈련 대상이 되는 바깥 구조가 `Model`이다. `Model`은 `Layer`와 비슷하되
추가로 훈련 및 직렬화를 위한 메서드가 있다.

이 모든 걸 모아서 전범위 예시로 만들어 보자. 변분 오토인코더(Variational
AutoEncoder, VAE)를 구현해서 MNIST 숫자로 훈련시킬 것이다.

그 VAE는 `Model`의 서브클래스이며 `Layer`의 서브클래스인 층들을 조합해서
만든다. 정칙화 손실(KL 발산)을 포함한다.


```python
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """(z_mean, z_log_var)를 가지고 숫자 인코딩 벡터 z를 만든다."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """MNIST 숫자를 튜플 (z_mean, z_log_var, z)로 사상한다."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """숫자 인코딩 벡터 z를 읽을 수 있는 숫자로 다시 변환한다."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """인코더와 디코더를 합쳐서 훈련 가능한 전구간 모델로 만든다."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL 발산 정칙화 손실 더하기
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

```

MNIST에 도는 간단한 훈련 루프를 작성해 보자.


```python
original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 2

# epochs 번 돌기
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # 데이터셋 배치들을 가지고 돌기
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # 복원 손실 계산
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # KLD 정칙화 손실 더하기

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
```

<div class="k-default-codeblock">
```
Start of epoch 0
step 0: mean loss = 0.3431
step 100: mean loss = 0.1273
step 200: mean loss = 0.1001
step 300: mean loss = 0.0897
step 400: mean loss = 0.0847
step 500: mean loss = 0.0812
step 600: mean loss = 0.0790
step 700: mean loss = 0.0774
step 800: mean loss = 0.0762
step 900: mean loss = 0.0751
Start of epoch 1
step 0: mean loss = 0.0748
step 100: mean loss = 0.0742
step 200: mean loss = 0.0737
step 300: mean loss = 0.0732
step 400: mean loss = 0.0728
step 500: mean loss = 0.0724
step 600: mean loss = 0.0721
step 700: mean loss = 0.0718
step 800: mean loss = 0.0716
step 900: mean loss = 0.0713

```
</div>
VAE가 `Model`의 서브클래스이기 때문에 훈련 루프가 내장돼 있다.
따라서 다음처럼 훈련시킬 수도 있다.


```python
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)
```

<div class="k-default-codeblock">
```
Epoch 1/2
938/938 [==============================] - 2s 2ms/step - loss: 0.0747
Epoch 2/2
938/938 [==============================] - 2s 2ms/step - loss: 0.0676

<keras.callbacks.History at 0x164668d90>

```
</div>
---
## 객체 지향 개발을 넘어서: 함수형 API

위 예가 지나치게 객체 지향 개발 느낌이라면
[함수형 API](/guides/functional_api/)를 이용해 모델을 만들 수도 있다.
어느 한 쪽을 택한다고 해서 다른 쪽 방식으로 작성된 요소들을 이용하지
못하는 게 아니라는 점을 잊지 말자. 언제든 섞어 쓸 수 있다.

예를 들어 아래의 함수형 API 예시에선 위 예에서 정의했던
`Sampling` 층을 재사용한다.


```python
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# 인코더 모델 정의
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# 디코더 모델 정의
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# VAE 모델 정의
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# KL 발산 정칙화 손실 더하기
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# 훈련
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
```

<div class="k-default-codeblock">
```
Epoch 1/3
938/938 [==============================] - 2s 1ms/step - loss: 0.0746
Epoch 2/3
938/938 [==============================] - 1s 1ms/step - loss: 0.0676
Epoch 3/3
938/938 [==============================] - 1s 1ms/step - loss: 0.0676

<keras.callbacks.History at 0x16469fc50>

```
</div>
더 많은 내용을 알고 싶으면 [함수형 API 안내서](/guides/functional_api/)를 읽어 보자.
