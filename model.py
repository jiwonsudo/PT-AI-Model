import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('./train_data.csv')

# 데이터의 전체 행 중 한국어 열을 선택해 숫자로 변환

# 한국어 데이터
exer_goal_ko_names = ['체중 감량', '근육 증가', '지구력 향상', '유연성 향상', '균형과 코어 강도 증가', '심폐 기능 향상', '스트레스 해소 및 정신 건강', '부상 예방 및 회복']
recommendation_ko_names = ['WPC', 'WPI', 'WPH', '카제인 프로틴', '혼합 단백질', '계란 프로틴', '식물성 프로틴', '소화효소포함 프로틴']

# 운동 목표 인덱스로 변환
df = df.replace(to_replace=exer_goal_ko_names, value=list(range(8)))
# 추천 결과 인덱스로 변환
df = df.replace(to_replace=recommendation_ko_names, value=list(range(8)))

# 마지막 열 드롭(추천 결과 프로틴 = recommendation_product)
df = df.drop(labels='recommendation_product', axis=1)

# 모델 구조제작
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
tfjs.converters.save_keras_model(model, './pt_ai_js')