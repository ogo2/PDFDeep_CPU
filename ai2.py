import tensorflow as tf

# Убедитесь, что устройство - CPU
with tf.device('/CPU:0'):
    model = YourModelClass()

    # Пример данных
    data = tf.random.normal([10, 3])

    # Пример тренировки
    for epoch in range(10):
        with tf.GradientTape() as tape:
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")