from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_split(csv_path, test_size=0.2, random_state=42):
    """
    Performs a stratified train/validation split based on the 'diagnosis' column.
    Ensures class distribution is preserved.
    """
    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["diagnosis"],
        random_state=random_state
    )

    return train_df, val_df