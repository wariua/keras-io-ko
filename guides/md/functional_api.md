# 함수형 API

**작성자:** [fchollet](https://twitter.com/fchollet)<br>
**생성 날짜:** 2019/03/01<br>
**최근 변경:** 2020/04/12<br>
**설명:** 함수형 API에 대한 안내서.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**Colab에서 보기**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/functional_api.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub 소스**](https://github.com/keras-team/keras-io/blob/master/guides/functional_api.py)



---
## 준비


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

---
## 소개

케라스 *함수형 API*는 `tf.keras.Sequential` API보다 유연하게 모델을 만들
수 있는 방식이다. 함수형 API를 이용하면 비선형 구조 모델이나 공유 층,
여러 입출력까지 다룰 수 있다.

핵심 발상은 딥 러닝 모델이 일반적으로 층들의 유향 무순환
그래프(directed acyclic graph, DAG)라는 것이다.
그래서 함수형 API는 *층들의 그래프*를 만드는 방법이다.

다음 모델을 생각해 보자.

<div class="k-default-codeblock">
```
(입력: 784차원 벡터)
       ↧
[Dense (64단위, relu 활성)]
       ↧
[Dense (64단위, relu 활성)]
       ↧
[Dense (10단위, softmax 활성)]
       ↧
(출력: 10개 분류에 대한 확률 분포 로짓)
```
</div>

3개 층으로 된 간단한 그래프다.
함수형 API를 써서 이 모델을 만들려면 먼저 입력 노드를 만들어야 한다.


```python
inputs = keras.Input(shape=(784,))
```

데이터 형태를 784차원 벡터로 설정했다.
표본의 형태만 지정하므로 배치 크기는 항상 생략한다.

예를 들어 `(32, 32, 3)` 형태인 이미지 입력이 있다면 다음처럼
작성하게 된다.


```python
# 시연용
img_inputs = keras.Input(shape=(32, 32, 3))
```

반환되는 `inputs`에는 모델에 넣어 줄 입력 데이터의 형태와 `dtype`에 대한
정보가 담겨 있다.
형태는 이렇다.


```python
inputs.shape
```




<div class="k-default-codeblock">
```
TensorShape([None, 784])

```
</div>
dtype은 이렇다.


```python
inputs.dtype
```




<div class="k-default-codeblock">
```
tf.float32

```
</div>
`inputs` 객체를 가지고 층을 호출해서 층 그래프에 새 노드를 추가한다.


```python
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```

위의 "층 호출" 동작은 "inputs"에서 새로운 층으로 화살표를 그리는 것과 비슷하다.
`dense` 층에 입력을 "전달"하고 출력으로 `x`를 얻는다.

층 그래프에 층을 더 추가하자.


```python
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

이제 층 그래프의 입력과 출력을 지정해서 `Model`을 만들 수 있다.


```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

모델 요약 정보를 확인해 보자.


```python
model.summary()
```

<div class="k-default-codeblock">
```
Model: "mnist_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 784)]             0         
                                                                 
 dense (Dense)               (None, 64)                50240     
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________

```
</div>
모델 구조를 그래프로 그릴 수도 있다.


```python
keras.utils.plot_model(model, "my_first_model.png")
```




    
![png](/img/guides/functional_api/functional_api_20_0.png)
    



원한다면 그래프를 그릴 때 각 층의 입력 및 출력 형태를 표시할 수도 있다.


```python
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```




    
![png](/img/guides/functional_api/functional_api_22_0.png)
    



이 그림과 코드가 거의 동일하다. 연결 화살표가 코드 버전에서는
호출 동작으로 바뀌어 있을 뿐이다.

"층 그래프"는 딥 러닝 모델을 쉽게 이해할 수 있는 심상이며
함수형 API를 이용해 층 그래프가 그대로 반영된 모델을 만들 수 있다.

---
## 훈련, 평가, 추론

함수형 API를 이용해 만든 모델에 대한 훈련과 평가, 추론은 `Sequential` 모델과
똑같은 방식으로 이뤄진다.

`Model` 클래스에는 내장 훈련 루프(`fit()` 메서드)와 내장 평가
루프(`evaluate()` 메서드)가 있다. 원한다면 [이 루프들을 손쉽게
원하는 대로 바꿔서](/guides/customizing_what_happens_in_fit/) 지도 학습
이상의 방식(예: [GAN](/examples/generative/dcgan_overriding_train_step/))으로
훈련 루틴을 구현할 수 있다.

아래에선 MNIST 이미지 데이터를 적재하고, 벡터 형태를 바꾸고,
(평가용 몫으로 성능을 관찰하면서) 데이터에 모델을 맞춘 다음,
테스트 데이터로 모델을 평가한다.


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

<div class="k-default-codeblock">
```
Epoch 1/2
750/750 [==============================] - 2s 2ms/step - loss: 0.3435 - accuracy: 0.9026 - val_loss: 0.1797 - val_accuracy: 0.9507
Epoch 2/2
750/750 [==============================] - 1s 2ms/step - loss: 0.1562 - accuracy: 0.9539 - val_loss: 0.1307 - val_accuracy: 0.9603
313/313 - 0s - loss: 0.1305 - accuracy: 0.9609 - 248ms/epoch - 793us/step
Test loss: 0.1305118203163147
Test accuracy: 0.9609000086784363

```
</div>
더 자세한 내용은 [훈련과 평가](/guides/training_with_built_in_methods/) 안내서를 보라.

---
## 저장과 직렬화

함수형 API를 써서 만든 모델을 저장하고 직렬화하는 건 `Sequential` 모델과
같은 방식으로 이뤄진다. 함수형 모델을 저장하는 표준 방식은 `model.save()`를
호출해서 전체 모델을 한 파일에 저장하는 것이다. 이후 그 모델을 만든 코드가
없어졌어도 그 파일을 가지고 동일한 모델을 다시 만들어 낼 수 있다.

저장된 파일에 다음 내용이 담긴다.

- 모델 구조
- (훈련 동안 학습한) 모델 가중치 값들
- (`compile`에 줬던) 모델 훈련 설정
- (훈련을 중지한 지점에서 재시작할 수 있게 해 주는) 최적화 방법 및 상태


```python
model.save("path_to_my_model")
del model
# 파일 내용만 가지고 똑같은 모델 다시 만들기
model = keras.models.load_model("path_to_my_model")
```

<div class="k-default-codeblock">
```
INFO:tensorflow:Assets written to: path_to_my_model/assets

```
</div>
자세한 내용은 모델 [직렬화와 저장](/guides/serialization_and_saving/)
안내서를 보라.

---
## 층 그래프를 여러 모델 정의에 사용하기

함수형 API에서는 층 그래프에 입력과 출력을 지정해서 모델을 만든다.
그렇다는 건 층 그래프 하나를 가지고 여러 모델을 만들어낼 수 있다는
뜻이다.

아래 예에선 같은 층들을 이용해 두 모델을 만든다. 이미지 입력을
16차원 벡터로 바꾸는 `encoder` 모델과 훈련을 위한 전범위 `autoencoder`
모델이다.


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

<div class="k-default-codeblock">
```
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                 
 global_max_pooling2d (Globa  (None, 16)               0         
 lMaxPooling2D)                                                  
                                                                 
=================================================================
Total params: 18,672
Trainable params: 18,672
Non-trainable params: 0
_________________________________________________________________
Model: "autoencoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                 
 conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
 )                                                               
                                                                 
 conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                 
 conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                 
 global_max_pooling2d (Globa  (None, 16)               0         
 lMaxPooling2D)                                                  
                                                                 
 reshape (Reshape)           (None, 4, 4, 1)           0         
                                                                 
 conv2d_transpose (Conv2DTra  (None, 6, 6, 16)         160       
 nspose)                                                         
                                                                 
 conv2d_transpose_1 (Conv2DT  (None, 8, 8, 32)         4640      
 ranspose)                                                       
                                                                 
 up_sampling2d (UpSampling2D  (None, 24, 24, 32)       0         
 )                                                               
                                                                 
 conv2d_transpose_2 (Conv2DT  (None, 26, 26, 16)       4624      
 ranspose)                                                       
                                                                 
 conv2d_transpose_3 (Conv2DT  (None, 28, 28, 1)        145       
 ranspose)                                                       
                                                                 
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________

```
</div>
디코딩 구조가 인코딩 구조와 엄밀하게 대칭인 것을 볼 수 있다.
따라서 출력 형태가 입력 형태인 `(28, 28, 1)`과 같다.

`Conv2D` 층의 반대가 `Conv2DTranspose` 층이고 `MaxPooling2D` 층의
반대가 `UpSampling2D` 층이다.

---
## 모든 모델은 층처럼 호출 가능하다

어떤 모델이든 마치 층인 것처럼 다룰 수 있다. 즉, `Input`을 가지고 호출하거나
다른 층의 출력을 가지고 호출할 수 있다. 모델을 호출할 때는 그 구조만
재사용하는 게 아니라 가중치도 재사용하게 된다.

실제 예를 보자. 다음은 오토인코더를 다른 방식으로 시도한 것이다. 인코더
모델과 디코더 모델을 만든 다음 두 번의 호출로 둘을 연결해서 오토인코더
모델을 얻는다.


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

<div class="k-default-codeblock">
```
Model: "encoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 original_img (InputLayer)   [(None, 28, 28, 1)]       0         
                                                                 
 conv2d_4 (Conv2D)           (None, 26, 26, 16)        160       
                                                                 
 conv2d_5 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 32)         0         
 2D)                                                             
                                                                 
 conv2d_6 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                 
 conv2d_7 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                 
 global_max_pooling2d_1 (Glo  (None, 16)               0         
 balMaxPooling2D)                                                
                                                                 
=================================================================
Total params: 18,672
Trainable params: 18,672
Non-trainable params: 0
_________________________________________________________________
Model: "decoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 encoded_img (InputLayer)    [(None, 16)]              0         
                                                                 
 reshape_1 (Reshape)         (None, 4, 4, 1)           0         
                                                                 
 conv2d_transpose_4 (Conv2DT  (None, 6, 6, 16)         160       
 ranspose)                                                       
                                                                 
 conv2d_transpose_5 (Conv2DT  (None, 8, 8, 32)         4640      
 ranspose)                                                       
                                                                 
 up_sampling2d_1 (UpSampling  (None, 24, 24, 32)       0         
 2D)                                                             
                                                                 
 conv2d_transpose_6 (Conv2DT  (None, 26, 26, 16)       4624      
 ranspose)                                                       
                                                                 
 conv2d_transpose_7 (Conv2DT  (None, 28, 28, 1)        145       
 ranspose)                                                       
                                                                 
=================================================================
Total params: 9,569
Trainable params: 9,569
Non-trainable params: 0
_________________________________________________________________
Model: "autoencoder"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                 
 encoder (Functional)        (None, 16)                18672     
                                                                 
 decoder (Functional)        (None, 28, 28, 1)         9569      
                                                                 
=================================================================
Total params: 28,241
Trainable params: 28,241
Non-trainable params: 0
_________________________________________________________________

```
</div>
보다시피 모델을 중첩시킬 수 있다. 즉, 모델 안에 하위 모델들을 넣을 수
있다. (모델이 층과 비슷하기 때문에 그렇다.)
모델 중첩을 흔히 이용하는 경우로 *앙상블 기법*이 있다.
예를 들어 다음처럼 여러 모델들을 한 모델로 합쳐서 예측 평균을 얻을 수 있다.


```python

def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

---
## 복잡한 그래프 구조 다루기

### 입력과 출력이 여럿인 모델

함수형 API를 쓰면 여러 입력과 출력을 쉽게 다룰 수 있다.
`Sequential` API로는 불가능하다.

예를 들어 고객의 이슈 티켓을 우선도에 따라 순위를 매겨서
정확한 부서로 보내는 시스템을 만든다고 하자.
그 모델에는 세 가지 입력이 있을 것이다.

- 티켓 제목 (텍스트 입력)
- 티켓 본문 텍스트 (텍스트 입력)
- 사용자가 추가한 태그 (범주형 입력)

그리고 두 가지 출력이 있을 것이다.

- 0에서 1 사이 우선도 점수 (스칼라 시그모이드 출력)
- 티켓을 처리해야 할 부서 (부서들의 집합에 대한 소프트맥스 출력)

함수형 API를 쓰면 몇 줄만으로 모델을 만들 수 있다.


```python
num_tags = 12  # 이슈 태그 종류
num_words = 10000  # 텍스트 데이터 전처리 시 얻는 어휘 크기
num_departments = 4  # 예측할 부서 수

title_input = keras.Input(
    shape=(None,), name="title"
)  # 가변 길이 int 열
body_input = keras.Input(shape=(None,), name="body")  # 가변 길이 int 열
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # `num_tags` 크기 이진 벡터들

# 제목의 각 단어를 64차원 벡터로 내장시키기
title_features = layers.Embedding(num_words, 64)(title_input)
# 텍스트의 각 단어를 64차원 벡터로 내장시키기
body_features = layers.Embedding(num_words, 64)(body_input)

# 제목의 내장 단어 열을 128차원 벡터 하나로 줄이기
title_features = layers.LSTM(128)(title_features)
# 본문의 내장 단어 열을 32차원 벡터 하나로 줄이기
body_features = layers.LSTM(32)(body_features)

# 모든 가용 피처들을 이어 붙여서 커다란 벡터 하나로 합치기
x = layers.concatenate([title_features, body_features, tags_input])

# 피처들 위에 우선도 예측을 위한 로지스틱 회귀 붙이기
priority_pred = layers.Dense(1, name="priority")(x)
# 피처들 위에 부서 분류기 붙이기
department_pred = layers.Dense(num_departments, name="department")(x)

# 우선도와 부서 모두를 예측하는 전범위 모델 생성하기
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)
```

모델 구조를 그려 보자.


```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```




    
![png](/img/guides/functional_api/functional_api_40_0.png)
    



모델을 컴파일할 때 출력마다 다른 손실 함수를 할당할 수 있다.
뿐만 아니라 손실마다 다른 가중치를 줘서 훈련 손실 총합에 기여하는
정도를 조절할 수도 있다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)
```

출력 층들의 이름이 다르기 때문에 손실 함수와 손실 가중치에
충 이름을 명시할 수도 있다.


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights={"priority": 1.0, "department": 0.2},
)
```

Numpy 배열들을 입력과 목표로 줘서 모델을 훈련시키자.


```python
# 가짜 입력 데이터
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# 가짜 목표 데이터
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)
```

<div class="k-default-codeblock">
```
Epoch 1/2
40/40 [==============================] - 3s 23ms/step - loss: 1.3256 - priority_loss: 0.7024 - department_loss: 3.1160
Epoch 2/2
40/40 [==============================] - 1s 25ms/step - loss: 1.2926 - priority_loss: 0.6976 - department_loss: 2.9749

<keras.callbacks.History at 0x1300d6110>

```
</div>
`Dataset` 객체로 fit을 호출할 때
`([title_data, body_data, tags_data], [priority_targets, dept_targets])`처럼
리스트들로 된 튜플을 주거나
`({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})`처럼
딕셔너리들로 된 튜플을 주어야 한다.

더 자세한 설명은 [훈련과 평가](/guides/training_with_built_in_methods/) 안내서를 보라.

### 간단한 ResNet 모델

함수형 API를 쓰면 입력과 출력이 여럿인 모델뿐 아니라
비선형 연결 구조도 쉽게 다룰 수 있다.
층들이 순차적으로 연결돼 있지 않은 그런 모델을
`Sequential` API로는 다룰 수 없다.

흔히 쓰는 경우로 잔여 연결이 있다.
CIFAR10에 대한 간단한 ResNet 모델을 만들어서 살펴보자.


```python
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()
```

<div class="k-default-codeblock">
```
Model: "toy_resnet"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 img (InputLayer)               [(None, 32, 32, 3)]  0           []                               
                                                                                                  
 conv2d_8 (Conv2D)              (None, 30, 30, 32)   896         ['img[0][0]']                    
                                                                                                  
 conv2d_9 (Conv2D)              (None, 28, 28, 64)   18496       ['conv2d_8[0][0]']               
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 9, 9, 64)    0           ['conv2d_9[0][0]']               
                                                                                                  
 conv2d_10 (Conv2D)             (None, 9, 9, 64)     36928       ['max_pooling2d_2[0][0]']        
                                                                                                  
 conv2d_11 (Conv2D)             (None, 9, 9, 64)     36928       ['conv2d_10[0][0]']              
                                                                                                  
 add (Add)                      (None, 9, 9, 64)     0           ['conv2d_11[0][0]',              
                                                                  'max_pooling2d_2[0][0]']        
                                                                                                  
 conv2d_12 (Conv2D)             (None, 9, 9, 64)     36928       ['add[0][0]']                    
                                                                                                  
 conv2d_13 (Conv2D)             (None, 9, 9, 64)     36928       ['conv2d_12[0][0]']              
                                                                                                  
 add_1 (Add)                    (None, 9, 9, 64)     0           ['conv2d_13[0][0]',              
                                                                  'add[0][0]']                    
                                                                                                  
 conv2d_14 (Conv2D)             (None, 7, 7, 64)     36928       ['add_1[0][0]']                  
                                                                                                  
 global_average_pooling2d (Glob  (None, 64)          0           ['conv2d_14[0][0]']              
 alAveragePooling2D)                                                                              
                                                                                                  
 dense_6 (Dense)                (None, 256)          16640       ['global_average_pooling2d[0][0]'
                                                                 ]                                
                                                                                                  
 dropout (Dropout)              (None, 256)          0           ['dense_6[0][0]']                
                                                                                                  
 dense_7 (Dense)                (None, 10)           2570        ['dropout[0][0]']                
                                                                                                  
==================================================================================================
Total params: 223,242
Trainable params: 223,242
Non-trainable params: 0
__________________________________________________________________________________________________

```
</div>
모델 구조를 그려 보자.


```python
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)
```




    
![png](/img/guides/functional_api/functional_api_51_0.png)
    



이제 모델을 훈련시키자.


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# Colab에서 도는 시간을 제한하기 위해 처음 1000개 샘플만 쓴다.
# 직접 전체 데이터셋으로 수렴할 때까지 훈련을 시켜 보자!
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)
```

<div class="k-default-codeblock">
```
13/13 [==============================] - 2s 98ms/step - loss: 2.3066 - acc: 0.1150 - val_loss: 2.2940 - val_acc: 0.1050

<keras.callbacks.History at 0x1305fee10>

```
</div>
---
## 공유 층

함수형 API를 쓰기 좋은 또 다른 경우는 *공유 층*이 있는 모델이다.
공유 층이란 같은 모델 안에서 여러 번 재사용하는 층 인스턴스를 뜻한다.
층 그래프의 여러 경로에 상응하는 피처들을 학습한다.

입력들이 비슷한 공간에서 올 때 (가령 두 텍스트의 어휘가 비슷할 때)
공유 층을 사용해 인코딩하는 경우가 많다.
공유 층을 쓰면 그 입력들 간에 정보를 공유할 수 있게 되고
더 적은 데이터로 모델을 훈련시키는 게 가능해진다.
입력들 중 하나에서 어떤 단어를 보게 되면
공유 층을 거치는 모든 입력들 처리에 활용할 수 있게 된다.

함수형 API에서 층을 공유하려면 같은 층 인스턴스를 여러 번 호출하면 된다.
예를 들어 다음에선 `Embedding` 층을 두 텍스트 입력에 걸쳐 공유한다.


```python
# 1000개 단어를 128차원 벡터로 사상시키는 Embedding
shared_embedding = layers.Embedding(1000, 128)

# 가변 길이 정수 열
text_input_a = keras.Input(shape=(None,), dtype="int32")

# 가변 길이 정수 열
text_input_b = keras.Input(shape=(None,), dtype="int32")

# 두 입력을 인코딩하는 데 같은 층을 재사용
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```

---
## 층 그래프 노드 추출해서 재사용하기

우리가 다루고 있는 층 그래프는 고정된 자료 구조이기 때문에
내부에 접근해서 내용을 조사할 수 있다. 그렇기 때문에
함수형 모델을 이미지로 그릴 수 있는 것이다.

또한 그렇기 때문에 중간 층들(그래프 "노드"들)의 활성에 접근하거나
중간 층을 다른 곳에서 재사용할 수 있다. 피처 추출 같은 작업에
매우 유용하다.

예를 살펴보자. 다음은 ImageNet으로 가중치를 미리 훈련시킨 VGG19 모델이다.


```python
vgg19 = tf.keras.applications.VGG19()
```

다음은 그래프 자료 구조를 질의해서 얻은 모델의
중간 활성들이다.


```python
features_list = [layer.output for layer in vgg19.layers]
```

이 피처들을 이용해 중간 층 활성 값들을 반환하는
피처 추출 모델을 만들어 보자.


```python
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
```

[신경망 화풍 전환](/examples/generative/neural_style_transfer/)
같은 작업에 특히 쓸모가 있다.

---
## 자체 제작 층으로 API 확장하기

`tf.keras`에는 다음과 같은 광범위한 내장 층들이 있다.

- 합성곱 층: `Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`
- 풀링 층: `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`
- RNN 층: `GRU`, `LSTM`, `ConvLSTM2D`
- `BatchNormalization`, `Dropout`, `Embedding` 등

하지만 원하는 게 없다면 얼마든 새로운 층을 만들어서 API를 확장할 수 있다.
모든 층은 `Layer`의 서브클래스이고 다음을 구현해야 한다.

- `call` 메서드. 층에서 하는 계산을 명시한다.
- `build` 메서드. 층의 가중치들을 만든다. (사실 `__init___`에서도
가중치를 만들 수 있기 때문에 이는 관행일 뿐이다.)

층을 처음부터 만드는 방법을 자세히 배우려면
[새 층과 모델](/guides/making_new_layers_and_models_via_subclassing)
안내서를 읽으면 된다.

다음은 `tf.keras.layers.Dense`를 간단하게 구현한 것이다.


```python

class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
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


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
```

자체 제작 층에서 직렬화를 지원하려면 층 인스턴스 생성자 인자들을
반환하는 `get_config` 메서드를 정의해 주면 된다.


```python

class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
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


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
config = model.get_config()

new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
```

선택적으로 클래스 메서드 `from_config(cls, config)`를 구현할 수 있는데,
설정 딕셔너리를 가지고 층 인스턴스를 다시 만들 때 그 메서드를 쓴다.
`from_config` 기본 구현은 다음과 같다.

```python
def from_config(cls, config):
  return cls(**config)
```

---
## 함수형 API를 써야 할 때

케라스 함수형 API를 써서 새 모델을 만드는 게 나을까,
아니면 `Model`의 서브클래스를 만드는 게 나을까?
일반적으로는 함수형 API가 더 상위 수준이어서 쓰기 쉽고 안전하며
서브클래스 모델에서 지원하지 않는 여러 특장점도 있다.

하지만 유향 비순환 층 그래프로는 쉽게 나타낼 수 없는 모델을
만들 때는 서브클래스 모델의 유연성이 빛을 발한다.
예를 들어 트리 RNN을 구현해야 한다면 함수형 API로는 불가능하므로
`Model` 서브클래스를 만들어야 할 것이다.

함수형 API와 서브클래스 모델의 차이를 더 깊이 살펴보고 싶다면
[What are Symbolic and Imperative APIs in TensorFlow 2.0?](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html)
글을 읽어 볼 수 있다.

### 함수형 API의 강점

(마찬가지로 자료 구조의 일종인) 순차형 모델에서도 다음 특성들이 참이지만
(자료 구조가 아니라 파이썬 바이트코드인) 서브클래스 모델에선 참이 아니다.

#### 간결하다

`super(MyClass, self).__init__(...)`도 없고 `def call(self, ...):` 같은 것도 없다.

다음을

```python
inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)
```

서브클래스 버전과 비교해 보자.

```python
class MLP(keras.Model):

  def __init__(self, **kwargs):
    super(MLP, self).__init__(**kwargs)
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

# 모델 인스턴스 만들기.
mlp = MLP()
# 모델 상태를 만들기 위해 필요함.
# 최소 한 번 호출할 때까지 모델은 상태를 가지고 있지 않다.
_ = mlp(tf.zeros((1, 32)))
```

#### 연결 그래프 정의와 동시에 모델 검증이 이뤄진다

함수형 API에선 (`Input`을 써서) 입력 사양(형태와 dtype)을 미리 정한다.
각 층을 호출할 때마다 입력 사양이 층에서 상정하고 있는 조건에 맞는지
확인해서 아니면 친절한 오류 메시지가 찍힌다.

그래서 함수형 API로 만들 수 있는 모든 모델은 돌아간다는 게 보장된다.
(수렴 관련 디버깅을 제외하고) 모든 디버깅이 실행 시점이 아니라
모델을 구성하는 동안 정적으로 이뤄진다. 컴파일러의 타입 검사와
비슷하다.

#### 함수형 모델은 그리거나 조사할 수 있다

모델을 그래프로 그릴 수 있고 그 그래프의 중간 노드들에 손쉽게 접근할 수 있다.
예를 들어 (앞서의 예시처럼) 중간 층의 활성을 추출해서 재사용할 수 있다.

```python
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
```

#### 함수형 모델은 직렬화하거나 복제할 수 있다

함수형 모델은 코드 조각이 아니라 일종의 자료 구조이므로 안전하게
직렬화해서 파일 하나에 저장할 수 있다. 그래서 원본 코드가 없더라도
똑같은 모델을 다시 만들어 낼 수 있다.
[직렬화와 저장 안내서](/guides/serialization_and_saving/) 참고.

서브클래스 모델을 직렬화하려면 모델 수준에서 `get_config()` 및
`from_config()` 메서드를 구현해야 한다.


### 함수형 API의 약점

#### 역동적인 구조를 지원하지 않는다

함수형 API에선 모델을 층들의 DAG로 다룬다.
대부분의 딥 러닝 구조에선 이게 참이지만 항상 그런 건 아니다.
예를 들어 재귀 망이나 트리 RNN에선 이 가정이 성립하지 않으므로
함수형 API로 구현할 수 없다.

---
## API 섞어 쓰기

함수형 API와 서브클래스 모델은 양단간에 한 쪽을 택해서
그것만 써야 하는 이분법적 선택지가 아니다.
`tf.keras` API의 모든 모델들은 `Sequential` 모델이든, 함수형 모델이든,
바닥부터 새로 작성한 서브클래스 모델이든 서로 상호작용할수 있다.

서브클래스 모델이나 서브클래스 층의 일부에서 언제든 함수형 모델이나
`Sequential` 모델을 이용할 수 있다.


```python
units = 32
timesteps = 10
input_dim = 5

# 함수형 모델 정의하기
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        # 앞서 정의한 함수형 모델
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)


rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, timesteps, input_dim)))
```

<div class="k-default-codeblock">
```
(1, 10, 32)

```
</div>
서브클래서 층이나 모델에서 다음 중 한 패턴의 `call` 메서드를
구현하고 있기만 하다면 함수형 API에서 사용할 수 있다.

- `call(self, inputs, **kwargs)` --
여기서 `inputs`는 텐서거나 텐서들을 담은 구조(예: 텐서들의 리스트)이고
`**kwargs`는 텐서 아닌 (입력 아닌) 인자들이다.
- `call(self, inputs, training=None, **kwargs)` --
여기서 `training`은 층이 훈련 모드로 동작해야 할지
추론 모드로 동작해야 할지 나타내는 불리언이다.
- `call(self, inputs, mask=None, **kwargs)` --
여기서 `mask`는 불리언 마스크 텐서다. (예를 들어 RNN에 유용하다.)
- `call(self, inputs, training=None, mask=None, **kwargs)` --
당연히 마스킹과 훈련 모드 동작 지정을 동시에 할 수도 있다.

거기 더해서 자체 제작 층이나 모델에서 `get_config` 메서드를 구현하고 있다면
그걸 이용하는 함수형 모델이 계속 직렬화 가능하고 복제 가능하게 된다.

다음은 바닥부터 새로 작성한 RNN을 함수형 모델에 사용하는 간단한 예시다.


```python
units = 32
timesteps = 10
input_dim = 5
batch_size = 16


class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)


# `CustomRNN` 내부 계산에서 (영 텐서 `state`를 만들 때) 고정된 배치 크기가
# 필요하기 때문에 `batch_shape`로 고정된 입력 배치 크기를 지정한다.
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, 10, 5)))
```
