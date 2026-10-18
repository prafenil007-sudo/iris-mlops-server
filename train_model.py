from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Load the Iris dataset included with scikit-learn
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Keep the same species proportions in the training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scale measurements, then train the classifier
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(X_train, y_train)

    # Evaluate the model on data it did not train on
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
        )
    )

    # Save the model and the information the Flask app will need
    artifact = {
        "model": model,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names,
        "accuracy": float(accuracy),
    }

    output_path = Path(__file__).parent / "model.pkl"
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()