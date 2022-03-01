# 케라스에 대해

케라스는 기계 학습 플랫폼 [텐서플로](https://github.com/tensorflow/tensorflow) 위에서 도는 기계 학습 API다.
파이썬으로 작성돼 있으며 빠른 실험을 가능케 하는 데 중점을 두고 개발되었다. *아이디어에서 결과까지 시간을 최소화하는 건 좋은 연구를 위한 열쇠다.*

케라스는...

- **단순하다**. 그러면서도 너무 단순하진 않다. 개발자의 *인지 부담*을 줄여서 정말 중요한 문제에 집중할 수 있게 해 준다.
- **유연하다**. *점진적 복잡성 노출* 원칙을 따른다. 단순한 작업은 빠르고 쉽게 할 수 있다.
그러면서도 앞서 배운 것에서 이어지는 명확한 경로를 통해 복잡한 작업들도 얼마든 *가능하다*.
- **강력하다**. 업계 수준의 성능과 확장성을 제공한다. 나사, 유튜브, 웨이모 등 조직과 회사에서 이용한다.

---

## 케라스와 텐서플로 2

[텐서플로 2](https://www.tensorflow.org/)는 전구간을 다루는 오픈 소스 기계 학습 플랫폼이다.
[미분 가능 프로그래밍](https://en.wikipedia.org/wiki/Differentiable_programming)을 위한 기반 계층이라고 생각하면 되는데, 다음 네 가지 특성이 결합돼 있다.

- 저수준 텐서 연산을 CPU, GPU, TPU에서 효율적으로 실행하기.
- 미분 가능한 임의 식의 경사 계산하기.
- 수백 대 GPU 클러스터 같은 많은 장치에서 계산 수행하기.
- 프로그램("그래프")을 서버나 브라우저, 모바일 장치, 임베디드 장치 같은 외부 런타임으로 내보내기.

케라스는 텐서플로 2의 상위 API다. 기계 학습 문제를 풀기 위한 생산성 높은 인터페이스이며 최신 딥 러닝에 집중한다.
빠르게 반복하면서 기계 학습 솔루션을 개발해서 내놓는 데 꼭 필요한 추상화와 구성 요소들을 제공한다.

케라스를 통해 엔지니어와 연구자들이 텐서플로 2의 확장성과 크로스 플랫폼 역량을
모두 활용할 수 있다. TPU나 커다란 GPU 클러스터에서 케라스를 돌릴 수도 있고
케라스 모델을 내보내서 브라우저나 모바일 장치에서 돌릴 수도 있다.

---

## 케라스 처음 써 보기

케라스의 핵심 자료 구조는 **층(layer)**과 **모델(model)**이다.
가장 간단한 모델 종류는 [`Sequential` 모델](/guides/sequential_model/)로, 층들을 차례로 쌓은 것이다.
더 복잡한 구조에는 [케라스 함수형 API](/guides/functional_api/)를 쓰는 게 좋은데, 층들의 그래프를 마음대로 구성할 수 있다. 아니면 [서브클래스를 통해 모델 전체를 바닥부터 작성할 수도 있다](/guides/making_new_layers_and_models_via_subclassing/).

다음이 `Sequential` 모델이다.

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

층을 쌓으려면 `.add()`만 하면 된다.

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

모델이 괜찮다 싶으면 `.compile()`로 학습 과정에 대한 설정을 하자.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요하다면 최적화를 더 자세하게 설정할 수 있다. 케라스의 철학은 간단한 건 간단히 할 수 있게 하되
필요시 사용자가 모든 걸 제어할 수도 있게 하는 것이다. (그리고 서브클래스를 통해 소스 코드를 쉽게 확장할 수 있는 점이 궁극적인 제어다.)

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```

이제 훈련 데이터를 배치 단위로 돌릴 수 있다.

```python
# x_train과 y_train은 Numpy 배열
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

테스트 손실과 지표를 한 줄로 평가하자.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

또는 새 데이터에 대한 예측을 하게 하자.

```python
classes = model.predict(x_test, batch_size=128)
```

이상이 케라스를 사용하는 가장 기초적인 방식이다.

하지만 케라스는 최첨단 연구 아이디어를 돌려 볼 수 있는 고도로 유연한 프레임워크이기도 하다.
케라스는 *점진적 복잡성 노출* 원칙을 따른다. 그래서 시작하기 쉬우면서도 조금씩 배워 나가기만 하면 복잡한 경우들까지 얼마든 다룰 수 있다.

위에서 몇 줄만으로 간단한 신경망을 훈련 및 평가할 수 있었던 것과 마찬가지로
케라스를 이용해 새로운 훈련 방식이나 특이한 모델 구조를 빠르게 개발할 수도 있다.
다음은 저수준 훈련 루프 예시인데, 케라스에 텐서플로의 `GradientTape`를 결합한 것이다.

```python
import tensorflow as tf

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

    # 가중치에 대한 손실 경사 값 얻기
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # 모델의 가중치 갱신
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

다음에서 케라스에 대한 더 깊이 있는 튜토리얼을 볼 수 있다.

- [엔지니어를 위한 케라스 소개](/getting_started/intro_to_keras_for_engineers/)
- [연구자를 위한 케라스 소개](/getting_started/intro_to_keras_for_researchers/)
- [개발자 안내서](/guides/)

---

## 설치 및 호환성

케라스는 텐서플로 2에 `tensorflow.keras`로 포함돼 있다.
케라스를 쓰려면 [텐서플로 2를 설치](https://www.tensorflow.org/install)하기만 하면 된다.

케라스/텐서플로는 다음과 호환된다.

- 파이썬 3.7–3.10
- 우분투 16.04 이상
- 윈도우 7 이상
- 맥OS 10.12.6 (Sierra) 이상


---

## 지원

다음은 질문을 하거나 개발 논의에 참여할 수 있는 곳들이다.

- [텐서플로 포럼](https://discuss.tensorflow.org/).
- [케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users).

또 [깃허브 이슈](https://github.com/keras-team/keras/issues)로(만) **버그 보고 및 기능 요청**을 올릴 수 있다.
꼭 [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 먼저 읽어 보자.

---

## 왜 이름이 케라스인가?


케라스(Keras, κέρας)는 그리스어로 *뿔*을 뜻한다. 그리스 및 라틴 문헌에서 온 문학적 이미지를 따온 것인데, 첫 등장은 *오디세이아*다. 거기서 꿈의 정령들(_오네이로이_)은 상아의 문으로 지상에 와서 거짓 환영으로 사람을 속이는 쪽과 뿔의 문으로 와서 앞으로의 미래를 알려 주는 쪽으로 나뉜다. κέρας (뿔) / κραίνω (실현), 그리고 ἐλέφας (상아) / ἐλεφαίρομαι (속이기) 단어들로 언어 유희를 하는 것이다.

ONEIROS(Open-ended Neuro-Electronic Intelligent Robot Operating System) 프로젝트 연구 활동 일부로 케라스가 처음 개발되었다.

>_"Oneiroi are beyond our unravelling - who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ 호메로스, 오디세이아 19. 562 ff (Shewring 번역).

---


