import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs

# 예제 데이터
data = {
    'height': [170, 175, 180, 165, 160],
    'weight': [65, 70, 75, 60, 55],
    'age': [25, 30, 35, 28, 22],
    'activity_level': [1, 2, 3, 2, 1],  # 1: 낮음, 2: 보통, 3: 높음
    'goal': [0, 1, 1, 0, 0],  # 0: 체중 감량, 1: 근육 증가
    'protein_type': [0, 1, 1, 0, 0]  # 0: WPC, 1: WPI
}

df = pd.DataFrame(data)

# 입력(X)과 출력(Y) 데이터 분리
X = df[['height', 'weight', 'age', 'activity_level', 'goal']].values
Y = df[['protein_type']].values

# 모델 생성
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 0(WPC, 유청함유) 또는 1(WPI, 유청분리) 분류
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X, Y, epochs=50, batch_size=2)

# 모델 저장 및 변환
model.save("protein_model")
tfjs.converters.save_keras_model(model, "tfjs_model")  # 변환된 모델 저장
