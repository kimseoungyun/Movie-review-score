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

### 2. Data Augmentation
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
성능을 고도화하기 위해 다음 전략들을 도입할 예정입니다.
1. **Pre-trained Embeddings:** 처음부터 학습하는 대신 `GloVe`나 `Word2Vec`을 사용하여 의미론적 벡터를 초기화.
2. **Attention Mechanism:** 문장 내에서 감정 판단에 결정적인 단어(Keyword)에 가중치를 두는 Attention 기법 도입.
3. **Transformer (BERT):** RNN 계열의 순차적 처리 한계를 넘기 위해 Transformer 기반의 모델 적용.
