import time
import tensorflow as tf

STEPS = 300
BATCH_SIZE = 64
IMAGE_SIZE = 128
CLASSES = 10

print("TensorFlow :", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("Aucun GPU détecté")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPU :", gpus[0])

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3)),
        tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(64, 3, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(256, 3, strides=2, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(CLASSES),
    ]
)

optimizer = tf.keras.optimizers.Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Création préalable des variables
model(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)))
optimizer.build(model.trainable_variables)


@tf.function(jit_compile=True)
def train_step(step):
    images = tf.random.stateless_normal(
        (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),
        seed=(step, 0),
    )
    labels = tf.random.stateless_uniform(
        (BATCH_SIZE,),
        seed=(step, 1),
        maxval=CLASSES,
        dtype=tf.int32,
    )

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_function(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print("Compilation et préchauffage XLA...")
train_step(tf.constant(0))

print(f"Test de charge : {STEPS} étapes")
start = time.perf_counter()

for step in range(1, STEPS + 1):
    loss = train_step(tf.constant(step))

    if step % 50 == 0:
        print(f"Étape {step:3d}/{STEPS} — loss : {loss.numpy():.4f}")

duration = time.perf_counter() - start
images_processed = STEPS * BATCH_SIZE

memory = tf.config.experimental.get_memory_info("GPU:0")

print("\n=== RÉSULTATS ===")
print(f"Durée          : {duration:.2f} s")
print(f"Débit          : {images_processed / duration:.1f} images/s")
print(f"Mémoire actuelle : {memory['current'] / 1024**3:.2f} Gio")
print(f"Pic mémoire      : {memory['peak'] / 1024**3:.2f} Gio")
print(
    "Périphérique     :",
    model.trainable_variables[0].value.device,
)
print("Test terminé avec succès.")
