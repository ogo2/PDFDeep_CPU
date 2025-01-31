import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

# Генерация случайных данных для примера
# В реальной задаче используйте свои данные
np.random.seed(42)
X_train = np.random.rand(1000, 20)  # 1000 примеров, 20 признаков
y_train = np.random.randint(2, size=(1000, 1))  # Бинарные метки (0 или 1)

X_val = np.random.rand(200, 20)  # 200 примеров для валидации
y_val = np.random.randint(2, size=(200, 1))

# Создание модели
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели и сохранение истории обучения
history = model.fit(X_train, y_train,
                    epochs=50,  # Количество эпох
                    batch_size=32,
                    validation_data=(X_val, y_val))

# Визуализация результатов обучения
def plot_training_history(history):
    # Построение графиков для функции потерь
    plt.figure(figsize=(12, 4))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

# Вызов функции для построения графиков
plot_training_history(history)