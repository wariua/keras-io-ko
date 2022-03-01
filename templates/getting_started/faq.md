# 케라스 FAQ

자주 묻는 케라스 질문 목록.

## 일반적인 질문들

- [케라스 모델을 (한 머신의) 여러 GPU에서 훈련시키려면?](#케라스-모델을-한-머신의-여러-gpu에서-훈련시키려면)
- [훈련을 여러 머신으로 분산시키려면?](#훈련을-여러-머신으로-분산시키려면)
- [TPU에서 케라스 모델을 훈련시키려면?](#tpu에서-케라스-모델을-훈련시키려면)
- [케라스 설정 파일이 어디에 저장되는가?](#케라스-설정-파일이-어디에-저장되는가)
- [케라스에서 하이퍼파라미터를 조정하려면?](#케라스에서-하이퍼파라미터를-조정하려면)
- [케라스로 개발 중 재연 가능한 결과를 얻으려면?](#케라스로-개발-중-재연-가능한-결과를-얻으려면)
- [모델을 저장할 수 있는 방법들은?](#모델을-저장할-수-있는-방법들은)
- [모델 저장을 위해 HDF5 내지 h5py를 설치하려면?](#모델-저장을-위해-hdf5-내지-h5py를-설치하려면)
- [케라스를 어떻게 인용하면 되나?](#케라스를-어떻게-인용하면-되나)

## 훈련 관련 질문들

- ["표본", "배치", "에포크"가 무슨 뜻인가?](#표본-배치-에포크가-무슨-뜻인가)
- [왜 훈련 때 손실이 테스트 때 손실보다 훨씬 높은가?](#왜-훈련-때-손실이-테스트-때-손실보다-훨씬-높은가)
- [메모리에 다 안 들어가는 데이터셋에 케라스를 쓰려면?](#메모리에-다-안-들어가는-데이터셋에-케라스를-쓰려면)
- [프로그램이 중단됐을 때 돌던 훈련이 복원되게 하려면?](#프로그램이-중단됐을-때-돌던-훈련이-복원되게-하려면)
- [검사 손실이 더는 줄지 않을 때 훈련을 중단하려면?](#검사-손실이-더는-줄지-않을-때-훈련을-중단하려면)
- [층들을 고정시키고 미세 조정을 하려면?](#층들을-고정시키고-미세-조정을-하려면)
- [`call()`의 `training` 인자와 `trainable` 속성의 차이는?](#call의-training-인자와-trainable-속성의-차이는)
- [`fit()`에서 평가용 몫을 어떻게 계산하는가?](#fit에서-평가용-몫을-어떻게-계산하는가)
- [`fit()`에서 훈련 동안 데이터를 뒤섞는가?](#fit에서-훈련-동안-데이터를-뒤섞는가)
- [`fit()`으로 훈련시킬 때 지표를 관찰하는 좋은 방법은?](#fit으로-훈련시킬-때-지표를-관찰하는-좋은-방법은)
- [`fit()`의 동작 방식을 바꿔야 한다면?](#fit의-동작-방식을-바꿔야-한다면)
- [혼합 정밀도로 모델을 훈련시키려면?](#혼합-정밀도로-모델을-훈련시키려면)
- [`Model`의 메서드 `predict()`와 `__call__()`의 차이는?](#model의-메서드-predict와-call의-차이는)

## 모델 관련 질문들

- [중간 층의 출력을 얻으려면? (피처 추출)](#중간-층의-출력을-얻으려면-피처-추출)
- [미리 훈련된 모델을 케라스에서 쓰려면?](#미리-훈련된-모델을-케라스에서-쓰려면)
- [상태형 RNN을 쓰려면?](#상태형-rnn을-쓰려면)


---

## 일반적인 질문들


### 케라스 모델을 (한 머신의) 여러 GPU에서 훈련시키려면?

한 모델을 여러 GPU에서 돌리는 방법으로 **데이터 병렬화** 방식과 **장치 병렬화** 방식이 있다.
대부분 경우에 필요한 건 아마 데이터 병렬화 방식일 것이다.


**1) 데이터 병렬화**

데이터 병렬화란 대상 모델을 장치마다 하나씩 복제한 다음 각 복제본을 이용해 입력 데이터의 다른 부분들을 처리하게 하는 것이다.

케라스 모델에서 데이터 병렬화를 하는 가장 좋은 방법은 `tf.distribute` API를 이용하는 것이다. [케라스에서 <code>tf.distribute</code> 사용하는 방법에 대한 안내](/guides/distributed_training/)를 꼭 읽어 보자.

요지는 다음과 같다.

a) "분산 전략" 객체를 만든다. 가령 가용 장치마다 모델을 복제하고 모델 상태를 동기화하는 `MirroredStrategy`를 택할 수 있다.

```python
strategy = tf.distribute.MirroredStrategy()
```

b) 모델을 만들고 그 전략의 스코프 안에서 컴파일한다.

```python
with strategy.scope():
    # 함수형, 서브클래스 등 어떤 모델 종류든 가능
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
```

모든 상태 변수 생성이 그 스코프 안에서 이뤄져야 한다.
즉, 뭔가 추가로 변수를 만드는 경우 그 스코프 안에서 하면 된다.

c) `tf.data.Dataset` 객체를 입력으로 해서 `fit()`을 호출한다. 자체 콜백을 포함한 모든 종류의 콜백이 분산 동작과 대략 호환된다.
이 호출은 새 변수를 만들지 않으므로 전략 스코프 안에 있을 필요가 없다.

```python
model.fit(train_dataset, epochs=12, callbacks=callbacks)
```


**2) 모델 병렬화**

모델 병렬화는란 같은 모델의 다른 부분들을 다른 장치에서 돌게 하는 것이다. 분기 구조처럼 병렬 구조가 포함된 모델에 잘 맞는다.

텐서플로 장치 스코프를 이용해 할 수 있다. 간단한 예는 이렇다.

```python
# LSTM을 공유해서 두 열을 병렬로 인코딩하는 모델
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 한 GPU에서 첫 번째 열 처리하기
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(input_a)
# 다른 GPU에서 다음 열 처리하기
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(input_b)

# CPU에서 결과물 이어 붙이기
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate(
        [encoded_a, encoded_b], axis=-1)
```

---

### 훈련을 여러 머신으로 분산시키려면?

텐서플로 2에선 모델을 어떻게 분산시킬지를 거의 신경쓰지 않고
코드를 작성할 수 있다.
로컬에서 잘 도는 코드가 있을 때, 원하는 하드웨어에 대응하는
분산 전략(`tf.distribute.Strategy`)을 추가하는 것만으로
다른 코드 변경 없이 여러 작업 장비나 가속 장치로 코드를
분산시킬 수 있다.

케라스 모델도 마찬가지다. `tf.distribute` 분산 전략을 추가해서
모델을 구성하고 컴파일하는 코드를 감싸 주기만 하면
`tf.distribute` 분산 전략에 따라 훈련이 분산돼서 수행된다.

(한 머신에서 여러 장치를 활용해 훈련시키는 게 아니라)
여러 머신 상에 분산해 훈련시키는 경우에는 `MultiWorkerMirroredStrategy`와
`ParameterServerStrategy`라는 두 가지 분산 전략을 이용할 수 있다.

- `tf.distribute.MultiWorkerMirroredStrategy`는 케라스 스타일 모델의 구성과
훈련에 이용할 수 있는 동기적 CPU/GPU 다중 작업자 솔루션을 구현하고 있다.
여러 복제본들 간에 동기적으로 경사를 줄인다.
- `tf.distribute.experimental.ParameterServerStrategy`는 비동기 CPU/GPU
다중 작업자 솔루션을 구현하고 있다. 매개변수 서버에 매개변수들이 저장되고
작업 장비들이 비동기적으로 매개변수 서버의 경사 정보를 갱신한다.

단일 머신 다중 장치 훈련에 비해 분산 훈련이 할 일이 좀 더 많다.
`ParameterServerStrategy` 방식에선 "작업 서버"와 "매개변수 서버"로 이뤄진
원격 머신들을 띄우고 각각에서 `tf.distribute.Server`를 돌려야 한다.
그 다음으로 "지휘" 머신에서 원하는 파이썬 프로그램을 실행하게 되는데,
클러스터의 다른 머신들과 통신하는 방법을 명시한 환경 변수 `TF_CONFIG`가 필요하다.
`MultiWorkerMirroredStrategy` 방식에선 지휘 머신과 작업 머신들에서
같은 프로그램을 돌리며, 마찬가지로 클러스터와 통신하는 방법을 명시한
`TF_CONFIG` 환경 변수를 쓴다. 그렇게 준비하고 난 후의 흐름은 단일 머신
훈련과 비슷하다. 주된 차이는 분산 전략으로 `ParameterServerStrategy`나
`MultiWorkerMirroredStrategy`를 쓰게 된다는 점이다.

다음에 유의해야 한다.

- 클러스터의 모든 작업 장비에서 데이터를 효율적으로 당겨갈 수 있도록
데이터셋을 준비해야 한다. (예를 들어 구글 클라우드에서 클러스터를 돌린다면
구글 클라우드 스토리지에 데이터를 올려 두는 게 좋다.)
- 훈련 동작에 장애 내성이 있어야 한다. (예를 들어
`keras.callbacks.BackupAndRestore` 콜백 이용하기)

아래에 기본 흐름을 보여 주는 두 가지 코드가 있다. CPU/GPU 다중 작업자 훈련에
대해 더 자세히 알고 싶으면
[다중 GPU 훈련과 분산 훈련](/guides/distributed_training/)을 보라.
TPU 훈련에 대해선
[TPU에서 케라스 모델을 훈련시키려면?](#tpu에서-케라스-모델을-훈련시키려면)
항목을 보라.

`ParameterServerStrategy` 방식:

```python
cluster_resolver = ...
if cluster_resolver.task_type in ("worker", "ps"):
  # tf.distribute.Server 시작하고 기다리기.
  ...
elif cluster_resolver.task_type == "evaluator":
  # (선택적) 보조 평가 서버 돌리기.
  ...

# 아니면 분산 전략을 이용해 훈련을 제어하는 지휘 서버다.
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver=...)
train_dataset = ...

with strategy.scope():
  model = tf.keras.Sequential([
      layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'],
      steps_per_execution=10)

model.fit(x=train_dataset, epochs=3, steps_per_epoch=100)
```

`MultiWorkerMirroredStrategy` 방식:

```python
# 기본적으로 `MultiWorkerMirroredStrategy`는 `TF_CONFIG`의 클러스터
# 정보와 공통 연산 통신 방식 "AUTO"를 사용한다.
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
train_dataset = get_training_dataset()
with strategy.scope():
  # 전략 스코프 안에서 모델을 정의하고 컴파일한다. 그러면 전략 종류에
  # 맞춰서 변수들이 올바르게 생성되고 분산되고 초기화된다.
  model = tf.keras.Sequential([
      layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
model.fit(x=train_dataset, epochs=3, steps_per_epoch=100)
```

---

### TPU에서 케라스 모델을 훈련시키려면?

TPU는 빠르고 효율적인 딥 러닝용 하드웨어 가속기다. 구글 클라우드에서 이용 가능한데,
Colab, AI Platform (ML Engine), Deep Learning VM(환경 변수 `TPU_NAME` 설정)을 통해 이용할 수 있다.

먼저 [TPU 사용 안내서](https://www.tensorflow.org/guide/tpu)를 읽어 보자. 요약하면 다음과 같다.

(가령 Colab에서 TPU 런타임을 선택해서) TPU 런타임에 연결한 다음 `TPUClusterResolver`로 TPU를 탐지하게 해야 한다. 그러면 모든 지원 플랫폼에서 TPU를 자동으로 탐지한다.

```python
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU 탐지
print('Running on TPU: ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('Replicas: ', strategy.num_replicas_in_sync)

with strategy.scope():
    # 모델 만들기.
    ...
```

초기 설정 후의 흐름은 단일 머신 다중 GPU 훈련과 비슷하다.
주된 차이는 분산 전략으로 `TPUStrategy`를 쓰게 된다는 점이다.

다음에 유의해야 한다.

- 미리 정해진 고정 형태로 배치들이 나오도록 데이터셋을 준비해야 한다. TPU 그래프는 고정된 형태의 입력만 처리할 수 있다.
- TPU가 계속 돌 수 있도록 빠르게 데이터를 읽을 수 있어야 한다. [TFRecord 형식](https://www.tensorflow.org/tutorials/load_data/tfrecord)으로 데이터를 저장하는 게 도움이 될 수 있다.
- TPU가 계속 돌게 하기 위해 그래프 실행마다 경사 하강 단계를 여러 번 돌리는 걸 고려하자. `compile()`의 `experimental_steps_per_execution` 인자를 통해 그렇게 할 수 있다. 작은 모델에서 성능이 상당히 올라가게 된다.

---

### 케라스 설정 파일이 어디에 저장되는가?

모든 케라스 데이터가 저장되는 기본 디렉터리는 다음 위치다.

`$HOME/.keras/`

예를 들어 내가 쓰는 맥북 프로에선 `/Users/fchollet/.keras/`다.

윈도우 사용자라면 `$HOME`을 `%USERPROFILE%`로 바꿔야 한다.

케라스에서 (가령 권한 문제로) 위 디렉터리를 만들 수 없는 경우에는 `/tmp/.keras/`를 대신 쓴다.

케라스 설정 파일은 JSON 파일이고 `$HOME/.keras/keras.json`에 위치한다. 기본 설정은 다음과 같다.

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

다음 필드가 있다.

- 이미지 처리 층과 유틸리티들에서 기본적으로 사용할 이미지 데이터 형식. (`channels_last` 또는 `channels_first`)
-  일부 연산에서 0으로 나누기를 방지하기 위해 사용하는 수치 퍼징 인자 `epsilon`.
- 기본 부동소수점 데이터 타입.
- 기본 백엔드. 과거의 흔적이다. 요즘은 항상 텐서플로다.

비슷하게 [`get_file()`](/utils/#get_file)로 내려받은 파일 같은 데이터셋 파일 캐시본을 기본적으로 `$HOME/.keras/datasets/`에 저장하며,
케라스 응용들의 모델 가중치 캐시 파일들을 기본적으로 `$HOME/.keras/models/`에 저장한다.


---

### 케라스에서 하이퍼파라미터를 조정하려면?


[KerasTuner](https://keras.io/keras_tuner/) 사용을 권한다.

---

### 케라스로 개발 중 재연 가능한 결과를 얻으려면?

모델 개발 과정에서 성능 변화가 실제 모델 내지 데이터의 변화 때문인지 아니면 새 난수 시드의 결과일 뿐인지 판단하기 위해서 매번 재연 가능한 결과를 얻을 수 있으면 좋은 경우가 종종 있다.

일단 프로그램 시작 전에 (프로그램 내에서는 안 됨) 환경 변수 `PYTHONHASHSEED`를 `0`으로 설정해야 한다. 파이썬 3.2.3 이상에서 특정 해시 기반 동작들(가령 set이나 dict의 항목 순서. 자세한 내용은 [파이썬 문서](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)나 [이슈 #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926) 참고)이 재연 가능하도록 하려면 필요하다. 다음처럼 파이썬 시작 때 환경 변수를 설정해 줄 수 있다.

```shell
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 재연 불가능한 해시 (파이썬 3.2.3+)
8127205062320133199
$ python3 test_hash.py                  # 재연 불가능한 해시 (파이썬 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 재연 가능한 해시
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 재연 가능한 해시
4883664951434749476
```

또한 GPU에서 돌릴 때 일부 연산의 출력이 비결정적인데, 특히 `tf.reduce_sum()`이 그렇다. GPU에서 여러 연산을 병렬로 돌리기 때문에 실행 순서가 항상 보장되진 않기 때문이다. 부동소수점의 정밀도에 한계가 있기 때문에 숫자 몇 개를 더할 때도 그 순서에 따라 살짝 다른 결과가 나올 수 있다. 비결정적인 연산들을 피하려고 시도할 수는 있지만 텐서플로에서 경사 계산을 위해 자동으로 그런 연산을 생성하기도 하므로 코드를 CPU에서 돌리는 게 훨씬 간단하다. 그러기 위해 환경 변수 `CUDA_VISIBLE_DEVICES`를 빈 문자열로 설정할 수 있다.

```shell
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py
```

아래 코드에서 재연 가능한 결과를 얻는 방법의 예를 볼 수 있다.

```python
import numpy as np
import tensorflow as tf
import random as python_random

# Numpy에서 생성하는 난수를 잘 정의된 초기 상태로 시작하기 위해
# 아래 코드가 필요하다.
np.random.seed(123)

# 파이썬 코어에서 생성하는 난수를 잘 정의된 초기 상태로 시작하기
# 위해 아래 코드가 필요하다.
python_random.seed(123)

# 아래의 set_seed()는 텐서플로 백엔드의 난수 생성 기능을
# 잘 정의된 초기 상태로 만든다.
# 자세한 내용은 다음 참고:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
tf.random.set_seed(1234)

# 나머지 코드...
```

위와 같이 하면 개별 초기화 코드에서 시드를 설정해 줄 필요가 없다.
위에서 설정한 시드들의 조합에 의해 초기화 코드의 시드들이 정해지기
때문이다.


---

### 모델을 저장할 수 있는 방법들은?

*주의: 케라스 모델 저장에 pickle이나 cPickle을 쓰는 걸 권장하지 않는다.*

**1) 모델 전체 (설정 + 가중치) 저장하기**

모델 전체 저장이란 다음 내용을 담은 파일을 만드는 것이다.

- 모델의 구조. 모델을 다시 만들어 낼 수 있게.
- 모델의 가중치
- 훈련 설정 (손실, 최적화)
- 최적화 상태. 정확히 멈춘 지점에서 훈련을 재개할 수 있게.

기본 형식이자 권장하는 형식은 텐서플로의 [SavedModel 형식](https://www.tensorflow.org/guide/saved_model)이다.
텐서플로 2.0 이상에선 `model.save(your_file_path)`라고만 하면 된다.

명확하게 하려면 `model.save(your_file_path, save_format='tf')`라고 할 수 있다.

케라스에 원래부터 있던 HDF5 기반 저장 형식도 여전히 지원한다.
HDF5 형식으로 모델을 저장하려면 `model.save(your_file_path, save_format='h5')`라고 하면 된다.
`your_file_path`가 `.h5`나 `.keras`로 끝나면 자동으로 이 방식을 쓴다.
`h5py` 설치 방법에 대해선 [모델 저장을 위해 HDF5 내지 h5py를 설치하려면?](#모델-저장을-위해-hdf5-내지-h5py를-설치하려면) 항목을 보면 된다.

어느 형식으로든 모델을 저장한 다음엔 `model = keras.models.load_model(your_file_path)`로 되살릴 수 있다.

**예:**

```python
from tensorflow.keras.models import load_model

model.save('my_model')  # HDF5 파일 'my_model.h5' 생성
del model  # 기존 모델 삭제

# 이전과 동일한 컴파일된 모델 반환
model = load_model('my_model')
```


**2) 가중치만 저장하기**


**모델의 가중치**를 저장해야 한다면 아래처럼 HDF5로 하면 된다.

```python
model.save_weights('my_model_weights.h5')
```

모델을 만드는 코드는 있다고 할 때 다음처럼 해서 저장했던 가중치를 *같은* 구조의 모델로 적재할 수 있다.

```python
model.load_weights('my_model_weights.h5')
```

가령 미세 조정이나 전이 학습을 위해 *다른* (일부 층이 공통인) 구조 모델로 가중치를 적재하려면 *층 이름으로* 적재하면 된다.

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

예:

```python
"""
원래 모델이 다음과 같다고 할 때,

model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))
model.add(Dense(3, name='dense_2'))
...
model.save_weights(fname)
"""

# 새 모델
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 적재됨
model.add(Dense(10, name='new_dense'))  # 적재 안 됨

# 처음 모델의 가중치를 적재한다. 첫 번째 층 dense_1에만 영향을 준다.
model.load_weights(fname, by_name=True)
```

`h5py` 설치 방법에 대해선 [모델 저장을 위해 HDF5 내지 h5py를 설치하려면?](#모델-저장을-위해-hdf5-내지-h5py를-설치하려면) 항목을 보면 된다.


**3) 설정만 저장하기 (직렬화)**


가중치나 훈련 설정은 필요 없고 **모델 구조**만 저장해야 한다면 다음처럼 하면 된다.

```python
# JSON으로 저장
json_string = model.to_json()
```

생성되는 JSON 파일은 사람이 읽을 수 있는 형식이고 필요하면 직접 편집할 수 있다.

이제 그 데이터를 가지고 새 모델을 만들 수 있다.

```python
# JSON으로 모델 재구축하기:
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string)
```


**4) 저장된 모델의 자체 제작 층 (또는 기타 자체 제작 객체) 처리**

적재하려는 모델에 자체 제작 층이나 기타 자체 제작 클래스 내지 함수가 있다면
`custom_objects` 인자를 통해 적재 동작부로 전해 줄 수 있다.

```python
from tensorflow.keras.models import load_model
# 모델에 "AttentionLayer" 클래스가 포함돼 있다면
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

아니면 [자체 제작 객체 스코프](/api/utils/serialization_utils/#customobjectscope-class)를 이용할 수도 있다.

```python
from tensorflow.keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

`load_model` 및 `model_from_json`에서도 같은 방식으로 자체 제작 객체 처리가 된다.

```python
from tensorflow.keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 모델 저장을 위해 HDF5 내지 h5py를 설치하려면?

케라스 모델을 HDF5 파일로 저장하기 위해 케라스에선 파이썬 패키지 h5py를 이용한다.
의존 패키지 중 하나이므로 기본적으로 설치돼 있을 것이다.
데비안 기반 배포판에선 `libhdf5`를 추가로 설치해야 할 것이다.

<div class="k-default-code-block">
```
sudo apt-get install libhdf5-serial-dev
```
</div>

h5py가 설치돼 있는지 잘 모르겠다면 파이썬 셸을 열어서 모듈을 적재해 보면 된다.

```
import h5py
```

오류 없이 임포트되면 설치된 것이다. 아니라면
[자세한 설치 방법](http://docs.h5py.org/en/latest/build.html)을 확인해 보자.



---

### 케라스를 어떻게 인용하면 되나?

케라스가 연구에 도움이 된다면 결과물에서 인용해 달라. 다음은 BibTex 항목 예시다.

<code style="color: gray;">
@misc{chollet2015keras,<br>
&nbsp;&nbsp;title={Keras},<br>
&nbsp;&nbsp;author={Chollet, Fran\c{c}ois and others},<br>
&nbsp;&nbsp;year={2015},<br>
&nbsp;&nbsp;howpublished={\url{https://keras.io}},<br>
}
</code>

---

## 훈련 관련 질문들


### "표본", "배치", "에포크"가 무슨 뜻인가?


다음은 케라스 `fit()`을 올바로 이용하기 위해 알아 둬야 할 정의들이다.

- **표본(sample)**: 데이터셋의 항목 하나. 예를 들어 합성곱 망에선 이미지 하나가 **표본**이다. 음성 인식 모델에선 녹음 데이터 하나가 **표본**이다.

- **배치(batch)**: *N*개 표본으로 된 집합. **배치** 내의 표본들은 독립적으로 병렬로 처리된다. 훈련 중 한 배치가 모델에 갱신을 한 번만 일으킨다. **배치**는 일반적으로 입력 한 개보다는 입력 데이터 전체의 분포에 더 가깝다. 배치가 커질수록 더 가까워진다. 하지만 배치를 처리하는 데 시간이 더 걸리고 갱신이 한 번만 이뤄진다는 것 역시 사실이다. 추론(평가/예측)에선 메모리가 허용하는 한 최대한 크게 배치를 잡는 걸 권한다. (일반적으로 배치가 클수록 평가/예측이 더 빨라진다.)

- **에포크(epoch)**: 훈련 과정을 구별되는 단계들로 분리하기 위해 임의로 나눈 것이며, 일반적으로 "데이터셋 전체를 한 번 도는 것"으로 정의된다. 로그 기록이나 주기적 평가에 쓸모가 있다.
케라스 모델의 `fit` 메서드에 `validation_data`나 `validation_split`을 쓰면 **에포크** 끝마다 평가를 돌리게 된다.
케라스에선 특별히 설계된 [콜백](/api/callbacks/)을 추가해서 **에포크** 끝에 돌게 할 수 있다. 예를 들어 학습률을 바꾸거나 모델 체크포인트를 저장하는 데 쓸 수 있다.

---

### 왜 훈련 때 손실이 테스트 때 손실보다 훨씬 높은가?


케라스 모델에는 훈련 모드와 테스트 모드가 있다. 테스트 때는 드롭아웃이나 L1/L2 가중치 정칙화 같은 메커니즘들을 끊다.
그 메커니즘들이 훈련 손실에는 반영되지만 테스트 손실에는 반영되지 않는다.

그리고 케라스가 훈련 때 표시하는 손실은 각 훈련 데이터 배치의 손실들을 **현재 에포크 전체에서** 평균한 값이다.
시간이 지나며 모델이 바뀌기 때문에 에포크 첫 번째 배치의 손실이 일반적으로 마지막 배치보다 높다.
그래서 에포크 단위 평균이 나빠질 수 있다.
반면 에포크의 테스트 손실은 에포크 마지막의 모델을 이용해 계산하므로 손실이 낮게 나온다.


---

### 메모리에 다 안 들어가는 데이터셋에 케라스를 쓰려면?

[`tf.data` API](https://www.tensorflow.org/guide/data)를 이용해 `tf.data.Dataset` 객체를 만들면 된다.
로컬 디스크, 분산 파일 시스템, GCS 등에서 데이터를 당겨 올 수 있고
다양한 데이터 변형을 효율적으로 적용할 수 있는 데이터 파이프라인 추상 계층이다.

예를 들어 [`tf.keras.preprocessing.image_dataset_from_directory`](https://keras.io/api/preprocessing/image/#imagedatasetfromdirectory-function)
유틸리티는 로컬 디렉터리에서 이미지 데이터를 읽는 데이터셋을 만든다.
비슷하게 [`tf.keras.preprocessing.text_dataset_from_directory`](https://keras.io/api/preprocessing/text/#textdatasetfromdirectory-function)
유틸리티는 로컬 디렉터리에서 텍스트 파일을 읽는 데이터셋을 만든다.

데이터셋 객체들을 바로 `fit()`에 줄 수도 있고 직접 만든 저수준 훈련 루프에서 순회할 수도 있다.

```python
model.fit(dataset, epochs=10, validation_data=val_dataset)
```

---

### 프로그램이 중단됐을 때 돌던 훈련이 복원되게 하려면?

돌던 훈련이 불시에 중단되더라도 복원될 수 있게 (장애 내성이 있게) 하려면
`tf.keras.callbacks.experimental.BackupAndRestore`를 이용하면 된다.
에포크 번호와 가중치를 포함한 훈련 진행 내용을 정기적으로 디스크로 저장하며
다음 번 `Model.fit()` 호출 때 다시 적재한다.

```python
import tensorflow as tf
from tensorflow import keras

class InterruptingCallback(keras.callbacks.Callback):
  """훈련을 일부러 중단시키기 위한 콜백"""
  def on_epoch_end(self, epoch, log=None):
    if epoch == 15:
      raise RuntimeError('Interruption')

model = keras.Sequential([keras.layers.Dense(10)])
optimizer = keras.optimizers.SGD()
model.compile(optimizer, loss="mse")

x = tf.random.uniform((24, 10))
y = tf.random.uniform((24,))
dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(2)

backup_callback = keras.callbacks.experimental.BackupAndRestore(
    backup_dir='/tmp/backup')
try:
  model.fit(dataset, epochs=20, steps_per_epoch=5, 
            callbacks=[backup_callback, InterruptingCallback()])
except RuntimeError:
  print('***Handling interruption***')
  # This continues at the epoch where it left off.
  model.fit(dataset, epochs=20, steps_per_epoch=5, 
            callbacks=[backup_callback])
```

더 자세한 건 [콜백 문서](/api/callbacks/) 참고.


---

### 검사 손실이 더는 줄지 않을 때 훈련을 중단하려면?


`EarlyStopping` 콜백을 쓸 수 있다.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

더 자세한 건 [콜백 문서](/api/callbacks/) 참고.

---

### 층들을 고정시키고 미세 조정을 하려면?

**`trainable` 속성 설정하기**

모든 층과 모델에는 `layer.trainable`이라는 불리언 속성이 있다.

```shell
>>> layer = Dense(3)
>>> layer.trainable
True
```

모든 층과 모델에서 `trainable` 속성을 (참 또는 거짓으로) 설정할 수 있다.
`False`로 설정하면 `layer.trainable_weights` 속성이 비게 된다.

```python
>>> layer = Dense(3)
>>> layer.build(input_shape=(3, 3)) # 층의 가중치 만들기
>>> layer.trainable
True
>>> layer.trainable_weights
[<tf.Variable 'kernel:0' shape=(3, 3) dtype=float32, numpy=
array([[...]], dtype=float32)>, <tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([...], dtype=float32)>]
>>> layer.trainable = False
>>> layer.trainable_weights
[]
```

한 층에서 `trainable` 속성을 설정하면 모든 자식 층들(`self.layers` 내용물)에도 재귀적으로 설정된다.


**1) `fit()`으로 훈련시킬 때:**

`fit()`으로 미세 조정을 하려면,

- 기반 모델을 만들어서 사전 훈련 가중치를 적재하고,
- 기반 모델을 고정시키고,
- 그 위에 훈련 가능한 층들을 추가하고,
- `compile()` 및 `fit()`을 호출하면 된다.

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 고정

assert model.layers[0].trainable_weights == []  # ResNet50Base에 훈련 가능한 가중치 없음
assert len(model.trainable_weights) == 2  # Dense 층의 편향과 커널

model.compile(...)
model.fit(...)  # ResNet50Base 제외하고 Dense만 훈련
```

함수형 API나 모델 서브클래스 API에서도 비슷한 과정을 따르면 된다.
`trainable` 값을 바꾼 *후에* `compile()`을 호출해야 바뀐 내용들이 적용된다는 점을 기억하자.
`compile()` 호출에서 모델 훈련 단계의 상태 값들이 고정된다.


**2) 자체 훈련 루프를 쓸 때:**

훈련 루프를 작성할 때 (`model.weights` 전체가 아니라)
`model.trainable_weights` 부분의 가중치만 갱신해야 한다.

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 고정

# 데이터셋 배치들을 가지고 돌기
for inputs, targets in dataset:
    # GradientTape 열기
    with tf.GradientTape() as tape:
        # 진행
        predictions = model(inputs)
        # 이 배치에 대한 손실 값 계산하기
        loss_value = loss_fn(targets, predictions)

    # *훈련 가능한* 가중치에 대한 손실 경사 값 얻기
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델 가중치 갱신
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```


**`trainable`과 `compile()`의 상호작용**

모델에 `compile()`을 호출한다는 건 그 모델의 동작 방식을 "고정"시키는 것이다.
즉, 모델을 컴파일할 때의 `trainable` 속성 값이 다시 `compile`을 호출할 때까지
모델이 살아 있는 동안 유지된다는 것이다. 따라서 `trainable` 값을 바꾸고서
그게 반영되게 하려면 꼭 모델에 `compile()`을 다시 호출해야 한다.

예를 들어 모델 A와 B가 있어서 일부 층들을 공유한다고 할 때,

- 모델 A를 컴파일했고,
- 공유 층들에서 `trainable` 속성 값을 바꿨고,
- 모델 B를 컴파일했다면,

모델 A와 B가 공유 층들에 대해 다른 `trainable` 값을 쓰게 된다.
기존의 GAN 구현 대부분에서 이 메커니즘이 핵심 역할을 한다.

```python
discriminator.compile(...)  # `discriminator`를 훈련시킬 때 `discriminator`의 가중치가 갱신됨
discriminator.trainable = False
gan.compile(...)  # `gan`을 훈련시킬 때 그 하위 모델인 `discriminator`가 갱신되지 않음
```


---

### `call()`의 `training` 인자와 `trainable` 속성의 차이는?


`call`의 불리언 인자 `training`은 호출이 추론 모드로 돌아야 하는지
훈련 모드로 돌아야 하는지를 결정한다. 예를 들어 훈련 모드에선
`Dropout` 층에서 임의 뉴런들을 제외하고 출력을 조정한다.
추론 모드에선 그 층에서 아무것도 하지 않는다. 예:

```python
y = Dropout(0.5)(x, training=True)  # 훈련 때와 추론 때 *모두* 제외 동작을 켠다
```

층의 불리언 속성인 `trainable`은 훈련 시 손실을 최소화하기 위해
층의 훈련 가능 가중치들을 갱신해야 하는지를 결정한다.
`layer.trainable`이 `False`로 설정돼 있으면 `layer.trainable_weights`가 항상 빈 리스트가 된다. 예:

```python
model = Sequential([
    ResNet50Base(input_shape=(32, 32, 3), weights='pretrained'),
    Dense(10),
])
model.layers[0].trainable = False  # ResNet50Base 고정

assert model.layers[0].trainable_weights == []  # ResNet50Base에 훈련 가능한 가중치 없음
assert len(model.trainable_weights) == 2  # Dense 층의 편향과 커널

model.compile(...)
model.fit(...)  # ResNet50Base 제외하고 Dense만 훈련
```

보다시피 "추론 모드 대 훈련 모드"와 "층 가중치 훈련 가능 여부"는 전혀 다른 개념이다.

훈련 동안 역전파를 통해 출력 조정 비율을 학습하는 dropout 층이 있다고 상상해 보자.
그 이름이 `AutoScaleDropout`이라고 하자.
그 층은 어떤 훈련 가능한 상태를 가지고 있을 것이고, 추론 때와 훈련 때 동작이 다를 것이다.
`trainable` 속성과 `training` 호출 인자는 독립적이기 때문에 다음처럼 할 수 있다.

```python
layer = AutoScaleDropout(0.5)

# 훈련 때와 추론 때 *모두* dropout 적용
# 그리고 훈련 때 출력 조정 비율 학습
y = layer(x, training=True)

assert len(layer.trainable_weights) == 1
```

```python
# 훈련 때와 추론 때 *모두* dropout 적용
# 출력 조정 비율은 *고정*

layer = AutoScaleDropout(0.5)
layer.trainable = False
y = layer(x, training=True)
```


***`BatchNormalization` 층의 특이 사항***

미세 조정용 모델의 고정부에 `BatchNormalization` 층이 있을 수 있다.

`BatchNormalization` 층의 이동 통계치를 그대로 고정시킬지 아니면 새 데이터에 따라 갱신할지는 오랜 논의 주제였다.
과거에는 `bn.trainable = False`라고 하면 역전파만 멈출 뿐이고
훈련 시 통계 갱신을 막지 않았다. 하지만 많은 테스트를 해 본 결과
우리는 미세 조정 때 이동 통계치를 고정시키는 게 *일반적으로* 낫다는 걸 알게 됐다.
**텐서플로 2.0부터는 `bn.trainable = False`라고 하면 *추가적으로*
층이 추론 모드로 돌게 된다.**

이런 동작은 `BatchNormalization`에만 해당한다.
다른 층들에선 가중치 훈련 가능 여부와 "추론 모드 대 훈련 모드"가 그대로 독립적이다.



---

### `fit()`에서 평가용 몫을 어떻게 계산하는가?


`model.fit`의 `validation_split` 인자를 가령 0.1로 설정하면 데이터의 *마지막 10%*가 검사용 데이터가 되고, 0.25로 설정하면 데이터의 마지막 25%가 된다. 평가용 몫을 뽑아내기 전에 데이터를 섞지 않으며, 따라서 검사용 데이터는 말 그대로 입력 데이터의 *마지막* x% 표본들이다.

(한 `fit` 호출 내의) 모든 에포크에 같은 검사용 세트가 쓰인다.

데이터를 Numpy 배열로 줄 때만 `validation_split` 옵션을 이용할 수 있다. (색인이 불가능한 `tf.data.Datasets`에서는 불가능하다.)


---

### `fit()`에서 훈련 동안 데이터를 뒤섞는가?

데이터를 Numpy 배열로 주고 `model.fit()`의 `shuffle` 인자를 (기본값이기도 한) `True`로 설정하면 각 에포크에서 훈련 데이터를 전체적으로 뒤섞는다.

데이터를 `tf.data.Dataset` 객체로 주고 `model.fit()`의 `shuffle` 인자를 `True`로 설정하면 데이터셋을 지역적으로 뒤섞는다. (버퍼 뒤섞기)

`tf.data.Dataset` 객체를 사용할 때 (가령 `dataset = dataset.shuffle(buffer_size)` 호출로) 데이터를 미리 뒤섞는 게 좋다. 그러면 버퍼 크기를 제어할 수 있다.

검사용 데이터는 절대 뒤섞지 않는다.


---

### `fit()`으로 훈련시킬 때 지표를 관찰하는 좋은 방법은?

`fit()` 호출에서 기본적으로 보여 주는 진행 막대를 통해 손실 값과 지표 값을 볼 수 있다.
하지만 콘솔에서 계속 바뀌는 아스키 숫자들을 쳐다보는 게 지표를 관찰하는 최적의 방법은 아니다.
우리가 권하는 건 [텐서보드](https://www.tensorflow.org/tensorboard)다.
훈련 및 평가 지표들을 멋진 그래프로 표시해 주며 훈련 동안 주기적으로 갱신해 준다.
브라우저를 통해 접근할 수 있다.

[`TensorBoard` 콜백](/api/callbacks/tensorboard/)을 통해 `fit()`에서 텐서보드를 이용할 수 있다.

---

### `fit()`의 동작 방식을 바꿔야 한다면?

두 가지 방법이 있다.

**1) `Model` 클래스의 서브클래스를 만들고 `train_step` (및 `test_step`) 메서드 오버라이드하기**

자체적인 갱신 규칙을 쓰고 싶지만 `fit()`에서 제공하는 (콜백이나 효율적인 단계 융합 같은) 기능들은 활용하고 싶을 때 좋은 방식이다.

이 패턴을 쓴다고 함수형 API로 모델을 만들지 못하는 게 아니다.
직접 만든 클래스로 `inputs`와 `outputs`가 있는 모델을 만들게 된다.
순차형 모델도 마찬가지다. `keras.Sequential`의 서브클래스를 만들고
`keras.Model`이 아니라 `train_step`을 오버라이드하게 된다.

아래 예는 자체 `train_step`을 쓰는 함수형 모델을 보여 준다.

```python
from tensorflow import keras
import tensorflow as tf
import numpy as np

class MyCustomModel(keras.Model):

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


# MyCustomModel 인스턴스 구성 및 컴파일
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# 하던 대로 `fit` 이용
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=10)
```

또한 표본 가중치 지원을 쉽게 추가할 수 있다.

```python
class MyCustomModel(keras.Model):

    def train_step(self, data):
        # 데이터 풀기. 모델과 `fit()`에 주는 내용물에 따라
        # 그 구조가 달라진다.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # 진행
            # 손실 값 계산
            # 손실 함수는 `compile()에서 설정한다.
            loss = self.compiled_loss(y, y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

        # 경사 계산
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # 가중치 갱신
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 지표 갱신
        # 지표는 `compile()`에서 설정한다.
        self.compiled_metrics.update_state(
            y, y_pred, sample_weight=sample_weight)

        # 지표 이름으로 현재 값 얻을 수 있는 딕셔너리 반환
        # (self.metrics로 추적하는) 손실을 포함한다.
        return {m.name: m.result() for m in self.metrics}


# MyCustomModel 인스턴스 구성 및 컴파일
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = MyCustomModel(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# 이제 sample_weight 인자를 쓸 수 있음
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=10)
```

마찬가지로 `test_step`을 오버라이드해서 평가 방식을 바꿀 수도 있다.

```python
class MyCustomModel(keras.Model):

    def test_step(self, data):
      # 데이터 풀기
      x, y = data
      # 예측 계산하기
      y_pred = self(x, training=False)
      # 손실 추적용 지표 갱신
      self.compiled_loss(
          y, y_pred, regularization_losses=self.losses)
      # 지표 갱신하기
      self.compiled_metrics.update_state(y, y_pred)
      # 지표 이름으로 현재 값 얻을 수 있는 딕셔너리 반환
      # (self.metrics로 추적하는) 손실을 포함한다.
      return {m.name: m.result() for m in self.metrics}
```

**2) 저수준 자체 훈련 루프 작성하기**

모든 세부 사항을 제어하고 싶을 때 좋은 방식이다. 하지만 다소 길어질 수 있다. 예:

```python
# 최적화 준비
optimizer = tf.keras.optimizers.Adam()
# 손실 함수 준비
loss_fn = tf.keras.losses.kl_divergence

# 데이터셋 배치들을 가지고 돌기
for inputs, targets in dataset:
    # GradientTape 열기
    with tf.GradientTape() as tape:
        # 진행
        predictions = model(inputs)
        # 이 배치에 대한 손실 값 계산하기
        loss_value = loss_fn(targets, predictions)

    # 가중치에 대한 손실 경사값 얻기
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델의 가중치 갱신하기
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

이 예시에는 진행 막대 표시나 콜백 호출, 지표 갱신 같은 여러 주요 기능들이 빠져 있다.
필요하면 직접 해 줘야 한다. 전혀 어렵지 않지만 일이 좀 된다.


---

### 혼합 정밀도로 모델을 훈련시키려면?

케라스에선 GPU 및 TPU에서 혼합 정밀도 훈련을 기본으로 지원한다.
[상세한 안내서](https://www.tensorflow.org/guide/keras/mixed_precision) 참고.

---

### `Model`의 메서드 `predict()`와 `__call__()`의 차이는?

[Deep Learning with Python 2판](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras) 내용으로 답해 보겠다.

> `y = model.predict(x)`와 `y = model(x)`는 (여기서 `x`는 입력 데이터 배열) 모두
> "`x`에 모델 돌려서 출력 `y` 얻기"를 뜻한다. 하지만 정확히 똑같은 건 아니다.

> `predict()`는 배치 단위로 데이터를 돌려서
> (실제로 `predict(x, batch_size=64)`라고 해서 배치 크기를 지정할 수 있음)
> 출력의 NumPy 값들을 뽑아낸다. 대략 다음과 동등하다.
```python
def predict(x):
    y_batches = []
    for x_batch in get_batches(x):
        y_batch = model(x).numpy()
        y_batches.append(y_batch)
    return np.concatenate(y_batches)
```
> 따라서 `predict()` 호출은 아주 큰 배열에도 잘 동작한다.
> 반면 `model(x)`는 메모리에서 동작이 이뤄지므로 그렇지 않다.

> 한편으로 `predict()`는 미분 가능하지 않다. 그래서 `GradientTape` 스코프에서
> 호출 시 경사를 얻을 수 없다.
> 모델 호출에서 경사값을 얻을 필요가 있을 때는 `model(x)`을 쓰는 게 좋고
> 출력 값만 필요할 때는 `predict()`를 쓰는 게 좋다.
> 달리 말해 (지금 우리처럼) 저수준 경사 하강 루프를 작성하고 있는 게
> 아니라면 항상 `predict()`를 쓰면 된다.

---

## 모델 관련 질문들


### 중간 층의 출력을 얻으려면? (피처 추출)

함수형 API와 순차형 API에서 어떤 층이 딱 한 번만 호출됐다면
`layer.output`과 `layer.input`을 통해 그 출력과 입력을 가져올 수 있다.
이를 이용하면 다음처럼 손쉽게 피처 추출 모델을 만들 수 있다.

```python
from tensorflow import keras
from tensorflow.keras import layers

model = Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(10),
])
extractor = keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])
features = extractor(data)
```

당연히 `Model`의 서브클래스면서 `call`을 오버라이드하는 모델에는 불가능하다.

다른 방식도 있다. 다음 예시에서 만드는 `Model`은 지정한 층의 출력을 반환한다.

```python
model = ...  # 원래 모델 만들기

layer_name = 'my_layer'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(data)
```

---

### 미리 훈련된 모델을 케라스에서 쓰려면?

[`keras.applications`에 있는 모델들](/api/applications/)이나 [텐서플로 허브](https://www.tensorflow.org/hub)에 있는 모델들을 활용할 수 있다.
텐서플로 허브와 케라스가 잘 연동된다.

---

### 상태형 RNN을 쓰려면?


RNN을 상태형으로 만든다는 건 각 배치의 표본들에 의한 상태가 다음 배치 표본들을 위한 초기 상태로 재사용한다는 뜻이다.

따라서 상태형 RNN을 쓸 때는 다음을 상정한다.

- 모든 배치의 표본 개수가 같다.
- `x1`과 `x2`가 이웃한 배치일 때 모든 `i`에 대해 표본 `x2[i]`가 `x1[i]`의 후속 열이다.

RNN의 상태 유지 특성을 이용하려면:

- 모델 첫 번째 층에 `batch_size` 인자를 줘서 배치 크기를 명시적으로 지정해야 한다. 가령 단계당 16개 피처가 있고 10단계 열로 된 표본 32개로 배치가 이뤄져 있다면 `batch_size=32`.
- RNN 층(들)에 `stateful=True`를 설정해야 한다.
- `fit()` 호출 시 `shuffle=False`를 지정해야 한다.

누적된 상태를 재설정하려면:

- 모든 층들의 상태를 재설정하려면 `model.reset_states()` 사용
- 특정 상태형 RNN 층의 상태를 재설정하려면 `layer.reset_states()` 사용

예:

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

x = np.random.random((32, 21, 16))  # 입력 데이터. (32, 21, 16) 구조
# 이를 길이 10짜리 열로 모델에 넣어 줄 것이다.

model = keras.Sequential()
model.add(layers.LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(layers.Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 처음 10개 단계를 가지고 11번째 단계를 예측하도록 망을 훈련
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 망의 상태가 바뀌었다. 그 다음 열을 넣어 줄 수 있다.
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# LSTM 층의 상태를 재설정:
model.reset_states()

# 이 경우 가능한 다른 방식:
model.layers[0].reset_states()
```

참고로 `predict`, `fit`, `train_on_batch` 등 메소드 *모두가* 모델에 있는 상태형 층의 상태를 갱신하게 된다. 그래서 훈련뿐 아니라 예측도 상태 기반 방식으로 할 수 있다.


---

