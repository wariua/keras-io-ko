CV_GUIDES_MASTER = {
    "path": "keras_cv/",
    "title": "KerasCV",
    "toc": True,
    "children": [
        {
            "path": "cut_mix_mix_up_and_rand_augment",
            "title": "CutMix, MixUp, and RandAugment image augmentation with KerasCV",
        },
        {
            "path": "retina_net_overview",
            "title": "Train an Object Detection Model on Pascal VOC 2007 using KerasCV",
        },
        {
            "path": "custom_image_augmentations",
            "title": "Custom Image Augmentations with BaseImageAugmentationLayer",
        },
        {"path": "coco_metrics", "title": "Using KerasCV COCO Metrics"},
    ],
}

NLP_GUIDES_MASTER = {
    "path": "keras_nlp/",
    "title": "KerasNLP",
    "toc": True,
    "children": [
        {
            "path": "transformer_pretraining",
            "title": "Pretraining a Transformer from scratch with KerasNLP",
        },
    ],
}

KT_GUIDES_MASTER = {
    "path": "keras_tuner/",
    "title": "하이퍼파라미터 조정",
    "toc": True,
    "children": [
        {
            "path": "getting_started",
            "title": "Getting started with KerasTuner",
        },
        {
            "path": "distributed_tuning",
            "title": "Distributed hyperparameter tuning with KerasTuner",
        },
        {
            "path": "custom_tuner",
            "title": "Tune hyperparameters in your custom training loop",
        },
        {
            "path": "visualize_tuning",
            "title": "Visualize the hyperparameter tuning process",
        },
        {
            "path": "tailor_the_search_space",
            "title": "Tailor the search space",
        },
    ],
}

GUIDES_MASTER = {
    "path": "guides/",
    "title": "개발자 안내서",
    "toc": True,
    "children": [
        {
            "path": "functional_api",
            "title": "함수형 API",
        },
        {
            "path": "sequential_model",
            "title": "순차형 모델",
        },
        {
            "path": "making_new_layers_and_models_via_subclassing",
            "title": "서브클래스로 새 층과 모델 만들기",
        },
        {
            "path": "training_with_built_in_methods",
            "title": "케라스 내장 메서드를 이용한 훈련과 평가",
        },
        {
            "path": "customizing_what_happens_in_fit",
            "title": "`fit()` 내부 동작 바꾸기",
        },
        {
            "path": "writing_a_training_loop_from_scratch",
            "title": "훈련 루프 바닥부터 작성하기",
        },
        {
            "path": "serialization_and_saving",
            "title": "직렬화와 저장",
        },
        {
            "path": "writing_your_own_callbacks",
            "title": "자체 콜백 작성하기",
        },
        # {
        #     'path': 'writing_your_own_metrics',
        #     'title': 'Writing your own Metrics',
        # },
        # {
        #     'path': 'writing_your_own_losses',
        #     'title': 'Writing your own Losses',
        # },
        {
            "path": "preprocessing_layers",
            "title": "전처리 층 이용하기",
        },
        {
            "path": "working_with_rnns",
            "title": "순환 신경망 이용하기",
        },
        {
            "path": "understanding_masking_and_padding",
            "title": "마스킹과 패딩 이해하기",
        },
        {
            "path": "distributed_training",
            "title": "다중 GPU 훈련과 분산 훈련",
        },
        # {
        #     'path': 'tpu_training',
        #     'title': 'Training Keras models on TPU',
        # },
        {
            "path": "transfer_learning",
            "title": "전이 학습과 미세 조정",
        },
        # {
        #     'path': 'hyperparameter_optimization',
        #     'title': 'Hyperparameter optimization',
        # },
        KT_GUIDES_MASTER,
        CV_GUIDES_MASTER,
        NLP_GUIDES_MASTER,
        # TODO: mixed precision
    ],
}
