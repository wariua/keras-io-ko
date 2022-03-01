# 연구자를 위한 케라스 소개

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2020/04/01<br>
**최근 변경:** 2020/10/02<br>
**설명:** 딥 러닝 연구에 케라스와 텐서플로를 이용하기 위해 알아야 할 모든 것.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/intro_to_keras_for_researchers.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/intro_to_keras_for_researchers.py)



---
## 준비


```python
import tensorflow as tf
from tensorflow import keras
```

---
## 들어가며

기계 학습 연구자인가? NeurIPS에 논문을 출간하고 CV나 NLP 분야 최첨단에 있는가?
그렇다면 이 안내서에서 케라스 및 텐서플로 API의 핵심 개념들을 접할 수 있다.

이 안내서에서 다음을 배우게 된다.

- 텐서플로의 텐서, 변수, 경사
- `Layer`의 서브클래스로 층 만들기
- 저수준 훈련 루프 작성하기
- 층에서 `add_loss()` 메서드를 통해 만든 손실 추적하기
- 저수준 훈련 루프에서 지표 추적하기
- `tf.function` 컴파일로 실행 속도 높이기
- 층들을 훈련 모드나 추론 모드로 실행하기
- 케라스 함수형 API

또한 두 가지 전구간 예시 연구(변분 오토인코더와 하이퍼네트워크)를 통해
케라스 API가 실제 어떻게 쓰이는지 보게 된다.

---
## 텐서

텐서플로는 미분 가능 프로그래밍을 위한 기반 계층이다. 그 핵심은 N차원
배열(텐서)을 조작하기 위한 프레임워크다. NumPy와 상당히 비슷하다.

하지만 NumPy와 텐서플로에는 핵심적인 차이가 있다.

- 텐서플로는 GPU와 TPU 같은 하드웨어 가속기를 활용할 수 있다.
- 텐서플로는 모든 미분 가능 텐서 식의 경사를 자동으로 계산할 수 있다.
- 텐서플로의 계산 동작을 한 머신의 여러 장치로, 또는 (각기 여러 장치를 가질 수
있는) 여러 머신들로 분산시킬 수 있다.

텐서플로의 핵심에 있는 객체인 텐서에 대해 살펴보자.

다음은 상수 텐서다.


```python
x = tf.constant([[5, 2], [1, 3]])
print(x)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[5 2]
 [1 3]], shape=(2, 2), dtype=int32)

```
</div>
`.numpy()`를 호출해서 그 값을 NumPy 배열로 얻을 수 있다.


```python
x.numpy()
```




<div class="k-default-codeblock">
```
array([[5, 2],
       [1, 3]], dtype=int32)

```
</div>
NumPy 배열과 마찬가지로 `dtype` 속성과 `shape` 속성을 갖추고 있다.


```python
print("dtype:", x.dtype)
print("shape:", x.shape)
```

<div class="k-default-codeblock">
```
dtype: <dtype: 'int32'>
shape: (2, 2)

```
</div>
보통은 (`np.ones` 및 `np.zeros`처럼) `tf.ones` 및 `tf.zeros`를 통해 상수 센터를 만든다.


```python
print(tf.ones(shape=(2, 1)))
print(tf.zeros(shape=(2, 1)))
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[1.]
 [1.]], shape=(2, 1), dtype=float32)
tf.Tensor(
[[0.]
 [0.]], shape=(2, 1), dtype=float32)

```
</div>
난수 상수 텐서도 만들 수 있다.


```python
x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)

x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")

```

---
## 변수

변수란 (신경망 가중치 같은) 가변 상태를 저장하는 데 쓰는 특수한 텐서들이다.
어떤 초기값을 가지고 `Variable`을 만든다.


```python
initial_value = tf.random.normal(shape=(2, 2))
a = tf.Variable(initial_value)
print(a)

```

<div class="k-default-codeblock">
```
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[-1.7639292,  0.4263797],
       [-0.3954156, -0.6072024]], dtype=float32)>

```
</div>
`.assign(value)`, `.assign_add(increment)`, `.assign_sub(decrement)` 메서드를
이용해 `Variable`의 값을 갱신한다.


```python
new_value = tf.random.normal(shape=(2, 2))
a.assign(new_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j]

added_value = tf.random.normal(shape=(2, 2))
a.assign_add(added_value)
for i in range(2):
    for j in range(2):
        assert a[i, j] == new_value[i, j] + added_value[i, j]
```

---
## 텐서플로에서 수학 계산하기

NumPy 경험이 있다면 텐서플로의 수학 계산이 익숙하게 느껴질 것이다.
주된 차이는 텐서플로 코드는 GPU와 TPU에서 돌 수 있다는 점이다.


```python
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

c = a + b
d = tf.square(c)
e = tf.exp(d)
```

---
## 경사

여기서도 NumPy와 큰 차이가 있다. 미분 가능한 모든 식의 경사를 자동으로 얻을 수 있다.

`GradientTape`를 열어서 `tape.watch()`를 통해 텐서 "감시"를 시작한 다음,
그 텐서를 쓰는 미분 가능한 식을 작성하기만 하면 된다.


```python
a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
    tape.watch(a)  # `a`에 대한 계산 이력 기록 시작
    c = tf.sqrt(tf.square(a) + tf.square(b))  # `a`를 써서 뭔가 수학 계산하기
    # `a`에 대한 `c`의 경사는?
    dc_da = tape.gradient(c, a)
    print(dc_da)

```

<div class="k-default-codeblock">
```
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)

```
</div>
기본적으로 변수들은 자동으로 감시하므로 직접 `watch`를 하지 않아도 된다.


```python
a = tf.Variable(a)

with tf.GradientTape() as tape:
    c = tf.sqrt(tf.square(a) + tf.square(b))
    dc_da = tape.gradient(c, a)
    print(dc_da)
```

<div class="k-default-codeblock">
```
tf.Tensor(
[[ 0.99851996 -0.56305575]
 [-0.99985445 -0.773933  ]], shape=(2, 2), dtype=float32)

```
</div>
테이프를 중첩시켜서 고차 도함수를 계산할 수도 있다.


```python
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print(d2c_da2)

```

<div class="k-default-codeblock">
```
tf.Tensor(
[[1.2510717e-03 4.4079739e-01]
 [2.1326542e-04 3.7843192e-01]], shape=(2, 2), dtype=float32)

```
</div>
---
## 케라스 층

텐서플로가 **미분 가능 프로그래밍을 위한 기반 계층**이며
텐서, 변수, 경사를 다룬다면
케라스는 **딥 러닝을 위한 사용자 인터페이스**이며
층, 모델, 최적화, 손실 함수, 지표 등을 다룬다.

케라스는 텐서플로의 고수준 API 역할을 한다.
즉, 텐서플로를 간편하고 생산적으로 만들어 준다.

`Layer` 클래스는 케라스의 기초적인 추상 계층이다.
상태(가중치)와 (call 메서드에 정의된) 몇 가지 연산들을 캡슐화한다.

간단한 층은 다음처럼 생겼다.


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

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

`Layer` 인스턴스를 파이썬 함수처럼 사용하게 된다.


```python
# 층 인스턴스 만들기
linear_layer = Linear(units=4, input_dim=2)

# 층을 함수처럼 다룰 수 있다.
# 데이터를 가지고 호출해 보자.
y = linear_layer(tf.ones((2, 2)))
assert y.shape == (2, 4)
```

(`__init__`에서 생성한) 가중치 변수들이 `weights` 프로퍼티를 통해
자동으로 추적된다.


```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

`Dense`와 `Conv2D`, `LSTM`부터 `Conv3DTranspose`와 `ConvLSTM2D`처럼
복잡한 것까지 다양한 내장 층들이 있다. 내장된 기능들을 똑똑하게
잘 이용하자.

---
## 층 가중치 만들기

`self.add_weight()` 메서드를 이용하면 가중치를 더 편하게 만들 수 있다.


```python

class Linear(keras.layers.Layer):
    """y = w.x + b"""

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


# 층 인스턴스 만들기
linear_layer = Linear(4)

# 이렇게 하면 `build(input_shape)`를 호출하고 가중치들을 만든다.
y = linear_layer(tf.ones((2, 2)))
```

---
## 층 경사

`GradientTape` 내에서 층을 호출해서 층 가중치의 경사를 자동으로 얻을 수 있다.
그 경사를 이용해 층 가중치를 갱신할 수 있는데, 수동으로 할 수도 있고 최적화
객체를 이용할 수도 있다. 필요하다면 경사 값을 변경한 다음 사용할 수도 있다.


```python
# 데이터셋 준비
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# (위에서 정의한) 우리 선형 층의 인스턴스를 10 유닛으로 생성
linear_layer = Linear(10)

# 정수 목표치를 받는 로지스틱 손실 함수 생성
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 최적화 객체 생성
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# 데이터셋 배치들을 가지고 돌기
for step, (x, y) in enumerate(dataset):

    # GradientTape 열기
    with tf.GradientTape() as tape:

        # 진행
        logits = linear_layer(x)

        # 이 배치에 대한 손실 값
        loss = loss_fn(y, logits)

    # 가중치에 대한 손실의 경사 얻기
    gradients = tape.gradient(loss, linear_layer.trainable_weights)

    # 우리 선형 층의 가중치 갱신
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))

    # 진행 기록
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 2.4605865478515625
Step: 100 Loss: 2.3112568855285645
Step: 200 Loss: 2.1920084953308105
Step: 300 Loss: 2.1255125999450684
Step: 400 Loss: 2.020744562149048
Step: 500 Loss: 2.060229539871216
Step: 600 Loss: 1.9214580059051514
Step: 700 Loss: 1.7613574266433716
Step: 800 Loss: 1.6828575134277344
Step: 900 Loss: 1.6320191621780396

```
</div>
---
## 훈련 가능 가중치와 훈련 불가능 가중치

층에서 만드는 가중치는 훈련 가능이거나 불가능이다. 각각 `trainable_weights`와
`non_trainable_weights`를 통해 노출된다. 다음은 훈련 불가능 가중치가 있는
층이다.


```python

class ComputeSum(keras.layers.Layer):
    """입력들의 합을 반환."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # 훈련 불가능 가중치 만들기
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy())  # [2. 2.]

y = my_sum(x)
print(y.numpy())  # [4. 4.]

assert my_sum.weights == [my_sum.total]
assert my_sum.non_trainable_weights == [my_sum.total]
assert my_sum.trainable_weights == []
```

<div class="k-default-codeblock">
```
[2. 2.]
[4. 4.]

```
</div>
---
## 층을 담은 층

층들을 재귀적으로 중첩시켜서 더 큰 연산 단위를 만들 수 있다.
각 층에서 하위 계층들의 (훈련 가능 및 훈련 불가능) 가중치들을 알아서 추적한다.


```python
# 앞서 `build` 메서드를 만들었던 Linear 클래스를
# 이용해 보자.


class MLP(keras.layers.Layer):
    """Linear 층들을 쌓은 것."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLP()

# `mlp` 객체를 처음 호출할 때 가중치들이 만들어진다.
y = mlp(tf.ones(shape=(3, 64)))

# 가중치들이 재귀적으로 추적된다.
assert len(mlp.weights) == 6
```

우리가 직접 만든 위의 MLP는 내장 기능을 다음처럼 사용한 것과 동등하다.


```python
mlp = keras.Sequential(
    [
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dense(10),
    ]
)
```

---
## 층에서 만든 손실 추적하기

진행 동안 층에서 `add_loss()` 메서드를 통해 손실을 만들 수 있다.
정칙화 손실에 특히 유용하다.
하위 층에서 만든 손실을 부모 층에서 재귀적으로 추적한다.

다음은 활성 정칙화 손실을 만드는 층이다.


```python

class ActivityRegularization(keras.layers.Layer):
    """활성 희소성으로 정칙화 손실을 만드는 층."""

    def __init__(self, rate=1e-2):
        super(ActivityRegularization, self).__init__()
        self.rate = rate

    def call(self, inputs):
        # `add_loss`를 사용해서 입력에 의해 정해지는 정칙화
        # 손실을 만든다.
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

```

이 층을 포함하는 모델에서 이 정칙화 손실을 추적하게 된다.


```python
# MLP 블록에서 손실 층을 사용해 보자.


class SparseMLP(keras.layers.Layer):
    """Linear 층들에 희소성 정칙화 손실 추가."""

    def __init__(self):
        super(SparseMLP, self).__init__()
        self.linear_1 = Linear(32)
        self.regularization = ActivityRegularization(1e-2)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.regularization(x)
        return self.linear_3(x)


mlp = SparseMLP()
y = mlp(tf.ones((10, 10)))

print(mlp.losses)  # float32 스칼라 값 하나를 담은 리스트
```

<div class="k-default-codeblock">
```
[<tf.Tensor: shape=(), dtype=float32, numpy=0.21796302>]

```
</div>
진행 때마다 최상위 층에서 그 손실들을 초기화한다. 즉, 손실이 누적되지 않는다.
`layer.losses`는 항상 최근 진행에서 생긴 손실들만 담는다. 보통은 훈련 루프
작성 시 그 손실들을 더한 다음 경사를 계산하는 식으로 이용하게 된다.


```python
# losses는 *최근* 진행의 결과다.
mlp = SparseMLP()
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1
mlp(tf.ones((10, 10)))
assert len(mlp.losses) == 1  # 누적 안 됨

# 훈련 루프에서 이 손실들을 어떻게 이용하는지 보자.

# 데이터셋 준비
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# 새 MLP
mlp = SparseMLP()

# 손실과 최적화
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for step, (x, y) in enumerate(dataset):
    with tf.GradientTape() as tape:

        # 진행
        logits = mlp(x)

        # 이 배치에 대한 외부 손실 값
        loss = loss_fn(y, logits)

        # 진행 동안 생긴 손실 더하기
        loss += sum(mlp.losses)

        # 가중치에 대한 손실의 경사 얻기
        gradients = tape.gradient(loss, mlp.trainable_weights)

    # 우리 선형 층의 가중치 갱신
    optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))

    # 진행 기록
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 6.307978630065918
Step: 100 Loss: 2.5283541679382324
Step: 200 Loss: 2.4068050384521484
Step: 300 Loss: 2.3749840259552
Step: 400 Loss: 2.34563946723938
Step: 500 Loss: 2.3380157947540283
Step: 600 Loss: 2.3201656341552734
Step: 700 Loss: 2.3250539302825928
Step: 800 Loss: 2.344613790512085
Step: 900 Loss: 2.3183579444885254

```
</div>
---
## 훈련 지표 추적하기

케라스는 `tf.keras.metrics.AUC`나 `tf.keras.metrics.PrecisionAtRecall` 같은
다양한 내장 지표들을 제공한다. 또한 코드 몇 줄로 손쉽게 원하는 지표를
만들 수 있다.

자체 훈련 루프에서 지표를 쓰려면 다음처럼 하게 된다.

- 지표 객체를 만든다. 예: `metric = tf.keras.metrics.AUC()`
- 각 데이터 배치마다 `metric.update_state(targets, predictions)` 메서드를 호출한다.
- `metric.result()`를 통해 결과를 질의한다.
- 각 에포크 끝이나 평가 시작 때 `metric.reset_state()`를 통해 지표 상태를 초기화한다.

간단한 예를 보자.


```python
# 지표 객체 만들기
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# 층, 손실, 최적화 준비
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(2):
    # 데이터셋 배치들을 가지고 돌기
    for step, (x, y) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(x)
            # 이 배치에 대한 손실 값 계산
            loss_value = loss_fn(y, logits)

        # `accuracy` 지표의 상태 갱신
        accuracy.update_state(y, logits)

        # 손실 값을 최소화하도록 모델의 가중치 갱신
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # 지금까지의 정확도 값 기록
        if step % 200 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("Total running accuracy so far: %.3f" % accuracy.result())

    # 에포크 끝에서 지표 상태 초기화
    accuracy.reset_state()
```

<div class="k-default-codeblock">
```
Epoch: 0 Step: 0
Total running accuracy so far: 0.141
Epoch: 0 Step: 200
Total running accuracy so far: 0.751
Epoch: 0 Step: 400
Total running accuracy so far: 0.827
Epoch: 0 Step: 600
Total running accuracy so far: 0.859
Epoch: 0 Step: 800
Total running accuracy so far: 0.876
Epoch: 1 Step: 0
Total running accuracy so far: 0.938
Epoch: 1 Step: 200
Total running accuracy so far: 0.944
Epoch: 1 Step: 400
Total running accuracy so far: 0.944
Epoch: 1 Step: 600
Total running accuracy so far: 0.945
Epoch: 1 Step: 800
Total running accuracy so far: 0.945

```
</div>
그리고 `self.add_loss()` 메서드와 비슷하게 층의 `self.add_metric()` 메서드를
이용할 수 있다. 거기 어떤 값이든 주면 그 평균값이 추적된다. 층이나 모델에
`layer.reset_metrics()`를 호출해서 지표들의 값을 초기화할 수 있다.

또한 `keras.metrics.Metric`의 서브클래스를 만들어서 자체 지표를 정의할 수 있다.
위에서 호출했던 세 가지 함수를 오버라이드해야 한다.

- 통계 값 갱신하도록 `update_state()` 오버라이드
- 지표 값 반환하도록 `result()` 오버라이드
- 지표를 초기 상태로 재설정하도록 `reset_state()` 오버라이드

다음은 F1 점수 지표를 구현한 예시다. (표본 가중치 지정 가능.)


```python

class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", dtype="float32", threshold=0.5, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.threshold = 0.5
        self.true_positives = self.add_weight(
            name="tp", dtype=dtype, initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", dtype=dtype, initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", dtype=dtype, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater_equal(y_pred, self.threshold)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.cast(y_true & y_pred, self.dtype)
        false_positives = tf.cast(~y_true & y_pred, self.dtype)
        false_negatives = tf.cast(y_true & ~y_pred, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_positives *= sample_weight
            false_positives *= sample_weight
            false_negatives *= sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        return precision * recall * 2.0 / (precision + recall)

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

```

테스트를 해 보자.


```python
m = F1Score()
m.update_state([0, 1, 0, 0], [0.3, 0.5, 0.8, 0.9])
print("Intermediate result:", float(m.result()))

m.update_state([1, 1, 1, 1], [0.1, 0.7, 0.6, 0.0])
print("Final result:", float(m.result()))

```

<div class="k-default-codeblock">
```
Intermediate result: 0.5
Final result: 0.6000000238418579

```
</div>
---
## 함수 컴파일

디버깅에는 열심히 실행하는 모드가 좋다. 하지만 계산 동작을 정적 그래프로
컴파일하면 더 좋은 성능을 얻게 된다. 연구자에게 정적 그래프는 멋진 친구다.
어떤 함수든 `tf.function` 데코레이터로 감싸 주기만 하면 컴파일할 수 있다.


```python
# 층, 손실, 최적화 준비
model = keras.Sequential(
    [
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 훈련 단계 함수 만들기


@tf.function  # 빠르게!
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# 데이터셋 준비
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 2.291861057281494
Step: 100 Loss: 0.5378965735435486
Step: 200 Loss: 0.48008084297180176
Step: 300 Loss: 0.3359006941318512
Step: 400 Loss: 0.28147661685943604
Step: 500 Loss: 0.31419697403907776
Step: 600 Loss: 0.2735794484615326
Step: 700 Loss: 0.3001103401184082
Step: 800 Loss: 0.18827161192893982
Step: 900 Loss: 0.15798673033714294

```
</div>
---
## 훈련 모드와 추론 모드

`BatchNormalization` 층과 `Dropout` 층 같은 일부 층들은 훈련 때와 추론 때의
동작이 다르다. 그런 층들에선 `call` 메서드에 `training`이라는 (불리언) 인자를
제공하는 게 표준 관행이다.

`call`의 그 인자에 따라 내장 훈련 루프와 평가 루프를 켜서 층을 훈련 모드나
추론 모드로 올바르게 돌릴 수 있다.


```python

class Dropout(keras.layers.Layer):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class MLPWithDropout(keras.layers.Layer):
    def __init__(self):
        super(MLPWithDropout, self).__init__()
        self.linear_1 = Linear(32)
        self.dropout = Dropout(0.5)
        self.linear_3 = Linear(10)

    def call(self, inputs, training=None):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)
        return self.linear_3(x)


mlp = MLPWithDropout()
y_train = mlp(tf.ones((2, 2)), training=True)
y_test = mlp(tf.ones((2, 2)), training=False)
```

---
## 함수형 API를 통한 모델 구축

딥 러닝 모델을 만들기 위해 항상 객체 지향 프로그래밍을 해야 하는 건 아니다.
지금까지 본 모든 층들을 함수처럼 작성할 수도 있다. 이를 "함수형 API"라 한다.


```python
# `Input` 객체를 이용해 입력의 형태와 dtype을 기술한다.
# 말하자면 *타입 선언하기*에 해당한다.
# shape 인자는 표본의 크기다. 배치 크기는 포함하지 않는다.
# 함수형 API는 표본별 변형을 정의하는 데 집중한다.
# 아래 만드는 모델에서 자동으로 표본별 변형을 모아서 수행하므로
# 데이터 배치에 대해 모델을 호출할 수 있게 된다.
inputs = tf.keras.Input(shape=(16,), dtype="float32")

# 이 "타입" 객체를 가지고 층을 호출한다.
# 그러면 갱신된 타입(새로운 형태/dtype)이 반환된다.
x = Linear(32)(inputs)  # 앞서 정의했던 Linear 층 재사용
x = Dropout(0.5)(x)  # 앞서 정의했던 Dropout 층 재사용
outputs = Linear(10)(x)

# 입력과 출력을 지정해서 함수형 `Model`을 정의할 수 있다.
# 모델 자체도 하나의 층이다.
model = tf.keras.Model(inputs, outputs)

# 데이터를 가지고 호출하기 전에도 함수형 모델에는 이미 가중치가 있다.
# 그 입력 형태를 먼저 (`Input`에서) 정의해 줬기 때문이다.
assert len(model.weights) == 4

# 재미 삼아 적당한 데이터로 모델을 호출해 보자.
y = model(tf.ones((2, 16)))
assert y.shape == (2, 10)

# `__call__`에 `training` 인자를 줄 수 있다.
# (Dropout 층까지 전달될 것이다.)
y = model(tf.ones((2, 16)), training=True)
```

함수형 API는 보통 서브클래스 방식보다 간결하며, 그외 장점들(대략 타입 없는 객체 지향
개발 대비 타입 있는 함수형 언어가 제공하는 것과 같은 장점들)을 몇 가지 제공한다.
하지만 층의 DAG(유향 비순환 그래프)를 정의하는 데만 쓸 수 있다. 즉, 재귀적인 망은
Layer 서브클래스로 정의해야 한다.

[여기](/guides/functional_api/)서 함수형 API에 대해 더 배울 수 있다.

연구 작업 때는 객체 지향 모델과 함수형 모델을 적재적소에 섞어 쓰게 되는 일이
많을 것이다.

`Model` 클래스에는 훈련 및 평가 루프인 `fit()`, `predict()`, `evaluate()`도
내장돼 있다. (`compile()` 메서드를 통해 설정한다.) 그 내장 함수들을 통해
다음과 같은 내장 훈련 기능들을 이용할 수 있다.

* [콜백](/api/callbacks/). 일찍 멈추기, 모델 체크포인트, 텐서보드를 통한
훈련 상태 관찰에 콜백을 이용할 수 있다. 또한 필요시 [자체 콜백을
구현](/guides/writing_your_own_callbacks/)할 수도 있다.
* [분산 훈련](/guides/distributed_training/). `tf.distribute` API를 이용해
훈련을 여러 GPU, TPU, 또는 여러 머신으로 손쉽게 확장할 수 있다. 어떤 코드
변경도 필요치 않다.
* [단계 합치기](/api/models/model_training_apis/#compile-method).
`Model.compile()`의 `steps_per_execution` 인자를 이용해 `tf.function` 호출
한 번에 여러 배치를 처리할 수 있다. TPU에서 장치 사용률이 크게 향상된다.

자세히 들어가지는 않겠고 아래에 간단한 코드 예시가 있다. 내장 훈련 기능들을
활용해 위의 MNIST 예시를 구현한다.


```python
inputs = tf.keras.Input(shape=(784,), dtype="float32")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)
model = tf.keras.Model(inputs, outputs)

# `compile()`로 손실, 최적화, 지표 지정
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# 데이터셋으로 2 에포크 동안 모델 훈련시키기
model.fit(dataset, epochs=2)
model.predict(dataset)
model.evaluate(dataset)
```

<div class="k-default-codeblock">
```
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3958 - sparse_categorical_accuracy: 0.8872
Epoch 2/2
938/938 [==============================] - 1s 1ms/step - loss: 0.1916 - sparse_categorical_accuracy: 0.9447
938/938 [==============================] - 1s 798us/step - loss: 0.1729 - sparse_categorical_accuracy: 0.9485

[0.1728748232126236, 0.9484500288963318]

```
</div>
객체 지향 모델을 위한 내장 훈련 루프를 이용하고 싶다면 `Model`의 서브클래스를
만들면 된다. (`Layer`의 서브클래스를 만드는 것과 같은 식이다.)
`Model.train_step()`을 오버라이드하기만 하면 앞서 나열한 내장 기능들(콜백,
코드 추가 없는 분산 처리, 단계 합치기)을 이용하면서 `fit()` 안에서
일어나는 동작을 바꿀 수 있다.
마찬가지로 `test_step()`을 오버라이드해서 `evaluate()` 안에서 일어나는
동작을 바꿀 수 있고, `predict_step()`을 오버라이드해서 `predict()` 안에서
일어나는 동작을 바꿀 수 있다. 자세한 내용은
[이 안내서](/guides/customizing_what_happens_in_fit/)를 참고하라.


```python

class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def train_step(self, data):
        # 데이터 풀기. 모델과 `fit()`에 주는 내용물에 따라
        # 그 구조가 달라진다.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 진행
            loss = self.loss_fn(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        # 지표 갱신하기 (손실 추적 지표 포함)
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred)
        # 지표 이름으로 현재 값 얻을 수 있는 딕셔너리 반환
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    @property
    def metrics(self):
        # 우리 `Metric` 객체를 여기 나열해서 각 에포크 시작 때
        # `reset_states()`가 자동으로 호출되도록 한다.
        return [self.loss_tracker, self.accuracy]


inputs = tf.keras.Input(shape=(784,), dtype="float32")
x = keras.layers.Dense(32, activation="relu")(inputs)
x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)
model = CustomModel(inputs, outputs)
model.compile()
model.fit(dataset, epochs=2)
```

<div class="k-default-codeblock">
```
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3737 - accuracy: 0.8340
Epoch 2/2
938/938 [==============================] - 1s 946us/step - loss: 0.1934 - accuracy: 0.9405

<keras.callbacks.History at 0x15dfae110>

```
</div>
---
## 전구간 실험 예시 1: 변분 오토인코더

지금까지 다음을 배웠다.

- `Layer`는 (`__init__()`이나 `build()`에서 만드는) 상태와
(`call()`에서 정의하는) 연산을 담는다.
- 층을 다른 층으로 감싸서 더 큰 연산 블록을 새로 만들 수 있다.
- 동작을 마음대로 바꿀 수 있는 훈련 루프를 손쉽게 작성할 수 있다.
`GradientTape`를 열어서 테이프 스코프 안에서 모델을 호출한 다음
경사를 얻어서 최적화 객체를 통해 적용하면 된다.
- `@tf.function` 데코레이터를 이용해 훈련 루프 속도를 올릴 수 있다.
- 층에서 `self.add_loss()`를 통해 손실(보통 정칙화 손실)을 만들고 추적할
수 있다.

이 모든 걸 모아서 전범위 예시로 만들어 보자. 변분 오토인코더(Variational
AutoEncoder, VAE)를 구현해서 MNIST 숫자로 훈련시킬 것이다.

우리가 만들려는 VAE는 `Layer`의 서브클래스인 층들을 중첩 조합해서 만든
`Layer` 서브클래스다. 정칙화 손실(KL 발산)을 포함한다.

모델을 정의해 보자.

먼저 `Encoder` 클래스가 있다. `Sampling` 층을 이용해서 MNIST 숫자를
잠재 공간 튜플 `(z_mean, z_log_var, z)`로 사상한다.


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

    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

```

다음으로 `Decoder` 클래스가 있다. 확률적 잠재 공간 좌표를 다시 MNIST 숫자로 사상한다.


```python

class Decoder(layers.Layer):
    """숫자 인코딩 벡터 z를 읽을 수 있는 숫자로 다시 변환한다."""

    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation=tf.nn.relu)
        self.dense_output = layers.Dense(original_dim, activation=tf.nn.sigmoid)

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)

```

마지막으로 `VariationalAutoEncoder`에서 인코더와 디코더를 조합하고 `add_loss()`를
통해 KL 발산 정칙화 손실을 만든다.


```python

class VariationalAutoEncoder(layers.Layer):
    """인코더와 디코더를 합쳐서 훈련 가능한 전구간 모델로 만든다."""

    def __init__(self, original_dim, intermediate_dim=64, latent_dim=32, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # KL 발산 정칙화 손실 추가
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed

```

이제 훈련 루프를 작성해 보자. 훈련 단계를 `@tf.function`으로 꾸며서
아주 빠른 그래프 함수로 컴파일한다.


```python
# 우리 모델
vae = VariationalAutoEncoder(original_dim=784, intermediate_dim=64, latent_dim=32)

# 손실과 최적화
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 데이터셋 준비
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.shuffle(buffer_size=1024).batch(32)


@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        reconstructed = vae(x)  # 입력 복원 계산
        # 손실 계산
        loss = loss_fn(x, reconstructed)
        loss += sum(vae.losses)  # KLD 항 추가
    # VAE의 가중치 갱신
    grads = tape.gradient(loss, vae.trainable_weights)
    optimizer.apply_gradients(zip(grads, vae.trainable_weights))
    return loss


losses = []  # 손실을 계속 추적한다.
for step, x in enumerate(dataset):
    loss = training_step(x)
    # 진행 기록
    losses.append(float(loss))
    if step % 100 == 0:
        print("Step:", step, "Loss:", sum(losses) / len(losses))

    # 1000단계 후 멈춘다.
    # 수렴할 때까지 모델을 훈련시키는 건
    # 연습으로 남겨 둔다.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 0.3246927559375763
Step: 100 Loss: 0.12636583357459247
Step: 200 Loss: 0.099717023916802
Step: 300 Loss: 0.0896754782535507
Step: 400 Loss: 0.08474012454065896
Step: 500 Loss: 0.08153954131933981
Step: 600 Loss: 0.07914437327577349
Step: 700 Loss: 0.07779341802723738
Step: 800 Loss: 0.07658644887466406
Step: 900 Loss: 0.07564477964855325
Step: 1000 Loss: 0.07468595038671474

```
</div>
보다시피 이런 종류의 모델을 만들고 훈련시키는 게 케라스에선
식은 죽 먹기다.

그런데 위 코드가 좀 장황해 보일 수도 있다. 온갖 세부 사항들을 직접 다루고 있기
때문이다. 유연성은 최고지만 작성하기가 좀 번거롭다.

VAE의 함수형 API 버전은 어떻게 되는지 살펴보자.


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

# KL 발선 정칙화 손실 추가
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)
```

훨씬 간결하다.

한편 케라스의 `Model` 클래스에는 훈련 루프와 평가 루프(`fit()`과 `evaluate()`)가
내장돼 있다. 확인해 보자.


```python
# 손실과 최적화
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# 데이터셋 준비
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    x_train.reshape(60000, 784).astype("float32") / 255
)
dataset = dataset.map(lambda x: (x, x))  # x_train을 입력이자 목표로 사용
dataset = dataset.shuffle(buffer_size=1024).batch(32)

# 모델 훈련 설정
vae.compile(optimizer, loss=loss_fn)

# 실제로 모델 훈련시키
vae.fit(dataset, epochs=1)
```

<div class="k-default-codeblock">
```
1875/1875 [==============================] - 3s 1ms/step - loss: 0.0713

<keras.callbacks.History at 0x15e150f10>

```
</div>
함수형 API와 `fit`을 써서 (모델 정의와 훈련이) 65행에서 25행으로 줄어들었다.
이렇게 생산성을 올려주는 요소들을 제공하면서도, 앞선 저수준 훈련 루프에서처럼
모든 걸 직접 작성해서 온갖 세부 사항들을 완전히 제어할 수도 있게 하는 것이
케라스의 철학이다.

---
## 전구간 실험 예시 2: 하이퍼네트워크

다른 종류의 연구 실험을 살펴보자. 이번엔 하이퍼네트워크다.

핵심 발상은 작은 심층 신경망(하이퍼네트워크)을 이용해 더 큰 망(메인 네트워크)의
가중치를 생성한다는 것이다.

정말 단순한 하이퍼네트워크를 만들어 보자. 작은 2층 망을 사용해 더 큰 2층짜리 망의
가중치를 생성하게 할 것이다.


```python
import numpy as np

input_dim = 784
classes = 10

# 레이블 예측에 실제 사용할 메인 네트워크
main_network = keras.Sequential(
    [keras.layers.Dense(64, activation=tf.nn.relu), keras.layers.Dense(classes),]
)

# 자체 가중치를 만들 필요가 없으므로 층이 이미 만들어져 있는 것처럼 표시하자.
# 이렇게 하면 `main_network` 호출 시 새 변수들이 생기지 않는다.
for layer in main_network.layers:
    layer.built = True

# 생성할 가중치 계수의 개수다. 메인 네트워크의 각 층마다
# output_dim * input_dim + output_dim 개 계수가 필요하다.
num_weights_to_generate = (classes * 64 + classes) + (64 * input_dim + 64)

# `main_network`의 가중치를 생성하는 하이퍼네트워크
hypernetwork = keras.Sequential(
    [
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(num_weights_to_generate, activation=tf.nn.sigmoid),
    ]
)
```

다음이 훈련 루프다. 각 데이터 배치에 대해서,

- `hypernetwork`을 이용해 가중치 계수 배열 `weights_pred`를 생성하고,
- 그 계수들의 형태를 바꿔서 `main_network`에 맞는 커널 및 편향 텐서들로 만들고,
- `main_network`를 진행시켜서 실제 MNIST 예측을 계산하고,
- 최종 분류 손실을 최소화하도록 `hypernetwork` 가중치들에 대해 역전파를 돌린다.


```python
# 손실과 최적화
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# 데이터셋 준비
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)

# 이 실험에는 크기 1인 배치를 사용한다.
dataset = dataset.shuffle(buffer_size=1024).batch(1)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # 외부 모델의 가중치 예측
        weights_pred = hypernetwork(x)

        # 외부 모델 w 및 b의 형태에 맞게 만든다.
        # 0번 층 커널
        start_index = 0
        w0_shape = (input_dim, 64)
        w0_coeffs = weights_pred[:, start_index : start_index + np.prod(w0_shape)]
        w0 = tf.reshape(w0_coeffs, w0_shape)
        start_index += np.prod(w0_shape)
        # 0번 층 편향
        b0_shape = (64,)
        b0_coeffs = weights_pred[:, start_index : start_index + np.prod(b0_shape)]
        b0 = tf.reshape(b0_coeffs, b0_shape)
        start_index += np.prod(b0_shape)
        # 1번 층 커널
        w1_shape = (64, classes)
        w1_coeffs = weights_pred[:, start_index : start_index + np.prod(w1_shape)]
        w1 = tf.reshape(w1_coeffs, w1_shape)
        start_index += np.prod(w1_shape)
        # 1번 층 편향
        b1_shape = (classes,)
        b1_coeffs = weights_pred[:, start_index : start_index + np.prod(b1_shape)]
        b1 = tf.reshape(b1_coeffs, b1_shape)
        start_index += np.prod(b1_shape)

        # 가중치 예측치를 외부 모델의 가중치 변수로 설정
        main_network.layers[0].kernel = w0
        main_network.layers[0].bias = b0
        main_network.layers[1].kernel = w1
        main_network.layers[1].bias = b1

        # 외부 모델에서 추론
        preds = main_network(x)
        loss = loss_fn(y, preds)

    # 내부 모델만 훈련
    grads = tape.gradient(loss, hypernetwork.trainable_weights)
    optimizer.apply_gradients(zip(grads, hypernetwork.trainable_weights))
    return loss


losses = []  # 손실을 계속 추적한다.
for step, (x, y) in enumerate(dataset):
    loss = train_step(x, y)

    # 진행 기록
    losses.append(float(loss))
    if step % 100 == 0:
        print("Step:", step, "Loss:", sum(losses) / len(losses))

    # 1000단계 후 멈춘다.
    # 수렴할 때까지 모델을 훈련시키는 건
    # 연습으로 남겨 둔다.
    if step >= 1000:
        break
```

<div class="k-default-codeblock">
```
Step: 0 Loss: 1.3274627923965454
Step: 100 Loss: 2.5709669510326765
Step: 200 Loss: 2.2051062234700542
Step: 300 Loss: 2.0191424489686534
Step: 400 Loss: 1.8865989956417193
Step: 500 Loss: 1.7706833476604333
Step: 600 Loss: 1.6479115988951523
Step: 700 Loss: 1.603230944064981
Step: 800 Loss: 1.533307248778922
Step: 900 Loss: 1.513232192888781
Step: 1000 Loss: 1.4671869220568465

```
</div>
어떤 연구 발상이든 케라스를 이용하면 간명하게, 그리고 생산적으로 구현할 수 있다.
매일 (실험당 평균 20분이 걸린다 치고) 25개 아이디어를 시도해 보는 걸 상상해 보라!

케라스는 발상에서 결과까지 최대한 빨리 진행할 수 있도록 설계되었다. 그게
훌륭한 연구를 위한 열쇠라고 믿기 때문이다.

이 소개서가 도움이 되었기를 바란다. 여러분들이 케라스로 뭘 만들지 궁금하다.
