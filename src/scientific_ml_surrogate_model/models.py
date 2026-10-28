from tensorflow import keras
from tensorflow.keras import layers


def build_encoder_decoder(
    number_of_parameters: int = 7,
) -> keras.Model:
    """Construit le modèle encodeur-décodeur."""

    inputs = keras.Input(
        shape=(number_of_parameters,),
        name="physical_parameters",
    )

    # encodeur des paramètres physiques
    x = layers.Dense(
        64,
        activation="relu",
    )(inputs)

    x = layers.Dense(
        128,
        activation="relu",
    )(x)

    x = layers.Dense(
        4 * 4 * 64,
        activation="relu",
    )(x)

    x = layers.Reshape(
        (4, 4, 64)
    )(x)

    # décodeur spatial : 4 x 4 vers 8 x 8
    x = layers.UpSampling2D(
        size=2,
        interpolation="bilinear",
    )(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    # 8 x 8 vers 16 x 16
    x = layers.UpSampling2D(
        size=2,
        interpolation="bilinear",
    )(x)

    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    # 16 x 16 vers 32 x 32
    x = layers.UpSampling2D(
        size=2,
        interpolation="bilinear",
    )(x)

    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    # champ de température normalisé
    outputs = layers.Conv2D(
        filters=1,
        kernel_size=3,
        padding="same",
        activation="linear",
        name="temperature_field",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="parameter_encoder_decoder",
    )


def build_mlp_baseline(
    number_of_parameters: int = 7,
) -> keras.Model:
    """Construit un MLP simple pour servir de référence."""

    inputs = keras.Input(
        shape=(number_of_parameters,),
        name="physical_parameters",
    )

    x = layers.Dense(
        128,
        activation="relu",
    )(inputs)

    x = layers.Dense(
        256,
        activation="relu",
    )(x)

    x = layers.Dense(
        512,
        activation="relu",
    )(x)

    x = layers.Dense(
        32 * 32,
        activation="linear",
    )(x)

    outputs = layers.Reshape(
        (32, 32, 1),
        name="temperature_field",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mlp_baseline",
    )


def compile_model(
    model: keras.Model,
    learning_rate: float = 1.0e-3,
) -> None:
    """Configure l'optimiseur, la loss et les métriques."""

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.MeanAbsoluteError(
                name="mae"
            ),
            keras.metrics.RootMeanSquaredError(
                name="rmse"
            ),
        ],
    )


if __name__ == "__main__":
    model = build_encoder_decoder()
    compile_model(model)
    model.summary()