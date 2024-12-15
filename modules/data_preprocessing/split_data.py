from sklearn.model_selection import train_test_split


def split_data(filenames, labels):
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        filenames, labels, test_size=0.4, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
