import time
import tensorflow as tf

print("TensorFlow :", tf.__version__)
print("GPU détectés :", tf.config.list_physical_devices("GPU"))

size = 2048
a = tf.random.normal((size, size))
b = tf.random.normal((size, size))

# Premier calcul pour initialiser TensorFlow et le GPU
tf.matmul(a, b).numpy()

start = time.perf_counter()
result = tf.matmul(a, b)
result.numpy()
duration = time.perf_counter() - start

print("Périphérique utilisé :", result.device)
print(f"Multiplication {size}×{size} : {duration:.3f} seconde(s)")
