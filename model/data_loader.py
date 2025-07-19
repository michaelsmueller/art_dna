import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(csv_path, include_test=False):
    df = pd.read_csv(csv_path)

    # First split: 80% train+val / 20% test
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["genre"], random_state=42
    )

    # Second split: 80% train / 20% val
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df["genre"], random_state=42
    )

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory="raw_data/resized",
        x_col="filename",
        y_col="genre",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    val_gen = datagen.flow_from_dataframe(
        val_df,
        directory="raw_data/resized",
        x_col="filename",
        y_col="genre",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    if include_test:
        test_gen = datagen.flow_from_dataframe(
            test_df,
            directory="raw_data/resized",
            x_col="filename",
            y_col="genre",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=False,  # Don't shuffle test data
        )
        return train_gen, val_gen, test_gen

    return train_gen, val_gen
