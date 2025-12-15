# 🎬 Movie Review Sentiment Analysis using Bi-RNN & Bi-LSTM

영화 리뷰 텍스트 데이터를 분석하여 긍정/부정을 예측하는 감성 분석(Sentiment Analysis) 프로젝트입니다. 기본적인 RNN 모델의 한계를 극복하기 위해 **Bidirectional 구조**를 도입하고, 데이터 증강(Augmentation)을 통해 성능을 개선하는 과정을 담았습니다.

## 📌 Project Overview
* **Goal:** 영화 리뷰 텍스트의 긍정(1) / 부정(0) 이진 분류 (Binary Classification)
* **Dataset:** `movie_reviews.csv(리뷰별 점수)`, `X_train.txt (리뷰)`
* **Models:** Bi-RNN, Bi-LSTM
* **Framework:** PyTorch

## 🧪 Key Experiments & Challenges

### 1. The Limitation of Vanilla RNN
초기 실험에서 단방향(Unidirectional) RNN 모델을 사용했을 때, **심각한 정보 손실(Information Loss)** 문제가 발생했습니다.
* **현상:** 모델이 모든 데이터를 하나의 클래스로만 예측하는 편향 현상 발생 (Mode Collapse).
* **시도:** 하이퍼파라미터 튜닝 및 Loss Function 변경을 시도했으나 해결되지 않음.
* **해결:** **Bidirectional(양방향) 구조**로 모델을 교체하여 문맥의 앞뒤 정보를 모두 학습하게 함으로써 문제를 근본적으로 해결함.

### 2. ⚠️ Troubleshooting: The "Single Class Prediction" Issue
프로젝트 진행 중, 초기 단방향 RNN 모델이 **이진 분류 학습 시 모든 데이터를 하나의 클래스(All 0 or All 1)로만 예측하는 치명적인 편향(Bias) 현상**을 발견했습니다. 이를 해결하기 위해 다양한 기술적 시도를 진행했습니다.

1.  **Loss Function Engineering (Failed):**
    * 데이터 불균형 문제로 의심하여 `BCEWithLogitsLoss`의 `pos_weight` 파라미터를 조정해 보았으나 효과가 없었습니다.
    * 특정 클래스로 쏠리는 것을 방지하기 위해 별도의 Penalty Loss를 추가해 보았으나, 근본적인 해결책이 되지 못했습니다.

2.  **Architectural Solution (Solved):**
    * 문제의 원인을 '파라미터 설정'이 아닌 **'단방향 모델의 구조적 정보 손실(Information Bottleneck)'**로 재정의했습니다.
    * 긴 문장의 맥락을 끝까지 유지하지 못하는 단방향 RNN의 한계를 **Bi-directional (양방향) 구조**로 교체함으로써, 문맥 정보를 보존하고 편향 문제를 완전히 해결했습니다.

### 3. Data Augmentation
* 원본 데이터(약 2,000개)만으로는 학습이 불안정하고 과적합(Overfitting)이 발생함.
* **전처리 전략:** 리뷰 데이터를 문장/줄 단위로 나누어 재생성함으로써 학습 데이터셋의 볼륨을 늘림.
* **결과:** 원본 데이터 사용 대비 정확도 약 **13% 향상**.

## 🛠️ Model Architecture & Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `MAX_SEQUENCE` | 100 | 최대 문장 길이 (Padding/Truncation) |
| `BATCH_SIZE` | 512 | 병렬 처리를 위한 배치 크기 |
| `EMBEDDING_DIM` | 256 | 단어 벡터 차원 |
| `HIDDEN_DIM` | 256 | 은닉층 차원 |
| `N_LAYERS` | 2 | Stacked RNN Layer 개수 |
| `DROPOUT` | 0.5 | 과적합 방지 |
| `EPOCHS` | 100 | 총 학습 횟수 |
| `LEARNING_RATE` | 0.001 | Optimizer 학습률 |

## 📊 Experimental Results

세 가지 시나리오에 대한 비교 실험 결과입니다.

| Model Type | Data Strategy | Test Accuracy | Test Loss | Note |
| :---: | :---: | :---: | :---: | :--- |
| **Bi-RNN** | **Augmented (Split Lines)** | **66.86%** | **0.601** | **Best Performance** 🏆 |
| Bi-LSTM | Augmented (Split Lines) | 66.20% | 0.661 | RNN 대비 학습 속도가 느림 |
| Bi-RNN | Original (2,000 samples) | 53.91% | 0.772 | 데이터 부족으로 학습 실패 |

### Result Analysis
1. **Bi-Directional의 필수성:** 긴 문장(Long Sequence)에서 앞쪽 정보가 소실되는 RNN의 고질적 문제를 양방향 학습으로 보완하여 유의미한 학습이 가능해졌습니다.
2. **RNN vs LSTM:** 이론적으로는 LSTM이 장기 의존성(Long-term dependency) 문제에 더 강하지만, 이번 실험에서는 Bi-RNN이 근소하게 더 높은 성능을 보였습니다. 이는 데이터셋의 특성상 문장 구조가 아주 복잡하지 않거나, LSTM이 과적합되었을 가능성을 시사합니다.
3. **학습 속도:** Simple RNN 구조가 LSTM보다 연산량이 적어 학습 속도가 월등히 빨랐습니다.

---
<img width="863" height="547" alt="image" src="https://github.com/user-attachments/assets/236c58a8-9c94-42ae-ac28-75a9e191b4d1" />
<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/448cf5f0-bf5d-4d82-bca3-11ef21b8cdee" />
<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/387c4e1c-e2e5-4f64-a982-4ca82e81a002" />

`정정 : SentimentBiGRU -> SEntimentBiRNN (샘플 2000개 사용)`

---

## 🚀 Future Works
현재 약 66%인 정확도를 80% 이상으로 끌어올리기 위해, 다음의 단계별 고도화 전략을 계획하고 있습니다.

### 1. Data Processing & Embedding
* **Text Preprocessing:** 단순 줄바꿈 처리를 넘어 `NLTK`나 `Spacy`를 활용해 불용어(Stopwords)를 제거하고, 표제어 추출(Lemmatization)을 적용하여 데이터의 밀도를 높입니다.
* **Advanced Tokenization:** 공백 기반 분리 대신 `WordPiece`나 `BPE` 토크나이저를 도입하여 미등록 단어(OOV) 문제를 해결합니다.
* **Pre-trained Embeddings:** 데이터셋만으로 학습하는 대신, `GloVe`나 `FastText` 같은 사전 학습된 임베딩 벡터를 사용하여 초기 성능을 확보합니다.

### 2. Model Architecture Improvements
* **GRU (Gated Recurrent Unit):** LSTM 대비 파라미터가 적어 학습 효율이 좋은 GRU 모델을 적용하여 적은 데이터셋에서의 성능을 비교합니다.
* **Attention Mechanism:** Bi-LSTM의 출력에 Attention Layer를 결합하여, 모델이 문장 내 핵심 감정 키워드에 집중할 수 있도록 개선합니다.

### 3. State-of-the-Art (SOTA) Models
* **BERT Fine-tuning:** RNN 계열의 한계를 극복하기 위해 `Hugging Face Transformers`를 활용, `bert-base` 모델을 파인 튜닝(Fine-tuning)하여 성능을 극대화합니다.
