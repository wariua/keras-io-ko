# 케라스 생태계

케라스 프로젝트는 신경망 구축 및 훈련을 위한 핵심 API에 한정된 것이 아니다.
기계 학습 작업의 모든 단계를 포괄하는 광범위한 관련 작업들을 아우른다.

---

## KerasTuner

[KerasTuner 문서](/keras_tuner/) - [KerasTuner GitHub 저장소](https://github.com/keras-team/keras-tuner)

KerasTuner는 하이퍼파라미터 탐색의 어려움을 해결해 주는 쓰기 쉽고 확장성 좋은 하이퍼파라미터 최적화 프레임워크다. 실행문으로 정의하는 문법을 이용해 탐색 공간을 쉽게 설정한 다음 여러 탐색 알고리듬 중 하나를 이용해 모델에 최적인 하이퍼파라미터 값들을 찾을 수 있다. KerasTuner에는 베이즈 최적화, Hyperband, 무작위 탐색 알고리듬이 내장돼 있으며, 연구자가 새로운 탐색 알고리듬을 실험해 보기 위해 확장하는 게 쉽도록 설계돼 있다.

---

## KerasNLP

[KerasNLP 문서](/keras_nlp/) - [KerasNLP GitHub 저장소](https://github.com/keras-team/keras-nlp)

KerasNLP는 자연어 처리(Natural Language Processing, NLP) 모델을 만들기 위한
간단하면서 강력한 API다. KerasNLP는 표준 케라스 인터페이스(층, 지표)를 따르는
구성 모듈들을 제공하며, 이를 통해 빠르고 유연하게 작업을 진행할 수 있다.
응용 NLP 분야 엔지니어는 이 라이브러리를 활용해 최첨단이면서 제품 수준인
훈련 및 추론 파이프라인을 조립할 수 있다. KerasNLP는 케라스 팀이 직접
관리한다.

---

## AutoKeras

[AutoKeras 문서](https://autokeras.com/) - [AutoKeras GitHub 저장소](https://github.com/keras-team/autokeras)

AutoKeras는 케라스 기반의 AutoML 시스템이다. 텍사스 A&M 대학교의 [DATA 랩](http://faculty.cs.tamu.edu/xiahu/index.html)에서 개발했다.
AutoKeras의 목표는 누구나 기계 학습을 이용할 수 있게 하는 것이다. 몇 줄만으로 기계 학습 문제를 해결할 수 있는
[`ImageClassifier`](https://autokeras.com/tutorial/image_classification/)나
[`TextClassifier`](https://autokeras.com/tutorial/text_classification/) 같은
고수준 전범위 API들을 제공하며, 구조 탐색을 수행하기 위한
[유연한 요소들](https://autokeras.com/tutorial/customized/)도 제공한다.

---

## KerasCV

[KerasCV 문서](/keras_cv/) - [KerasCV GitHub 저장소](https://github.com/keras-team/keras-cv)

KerasCV는 응용 컴퓨터 비전(Computer Vision, CV) 엔지니어가 이용할 수 있는 구성 모듈들(층, 지표, 손실, 데이터 증대)의 저장소다. 이를 활용해 이미지 분류, 사물 탐지, 이미지 분할, 이미지 데이터 증대 같은 흔한 용도를 위한 훈련 및 추론 파이프라인을 제품 수준이면서 최첨단으로 빠르게 조립할 수 있다.

KerasCV는 케라스 API의 수평 방향 확장이라고 생각할 수 있다. 즉, 직접 만들었지만 코어 케라스에 추가하기에는 너무 특수한 케라스 객체들(층, 지표 등)의 모음이다. 하지만 다른 케라스 API와 같은 수준의 손질과 하위 호환성 보장이 이뤄지며 (TFAddons와 달리) 케라스 팀 자체에서 관리한다.

---

## 텐서플로 클라우드

구글의 케라스 팀에서 관리하는 [텐서플로 클라우드](http://github.com/tensorflow/cloud) 최소한의 노력으로 GCP에서 대규모 케라스 훈련 작업을 돌리는 걸 도와 주는 도구들의 모음이다.
클라우드에서 8개나 그 이상 CPU에 실험 모델을 돌려 보는 게 `model.fit()` 호출만큼이나 쉽다.

---

## 텐서플로.js

[텐서플로.js](https://www.tensorflow.org/js)는 텐서플로의 자바스크립트 런타임으로, 브라우저나 [Node.js](https://nodejs.org/en/) 서버에서 텐서플로 모델을 (훈련 방식과 추론 방식 모두) 돌릴 수 있다.
케라스 모델 적재를 기본적으로 지원하며, 브라우저에서 직접 케라스 모델을 미세 조정하거나 재훈련시킬 수 있다.


---

## 텐서플로 Lite

[텐서플로 Lite](https://www.tensorflow.org/lite)는 장치 상의 효율적 추론을 위한 런타임이며 케라스 모델을 기본적으로 지원한다.
안드로이드, iOS, 임베디드 장치에 모델을 올릴 수 있다.


---

## 모델 최적화 도구

[텐서플로 모델 최적화 도구](https://www.tensorflow.org/model_optimization)는 *훈련 후 가중치 양자화*와 *가지치기 결합 훈련*을 수행해서 추론 모델을 더 빠르고, 메모리를 덜 먹고, 전기를 덜 먹게 만들 수 있는 도구들의 모음이다.
케라스 모델을 기본적으로 지원하며 그 가지치기 API가 케라스 API를 직접 이용해 만들어져 있다.


---

## TFX 연동

TFX는 머신 러닝 제품 파이프라인을 배치하고 관리하는 전범위 플랫폼이다.
TFX는 [케라스 모델을 기본적으로 지원](https://www.tensorflow.org/tfx/guide/keras)한다.


---
