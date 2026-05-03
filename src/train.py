"""for training modules """
import joblib
from src.data_loader import load_data
from src.model import train_model, evaluate_model


def main():
    """ function for loading data """
    df = load_data(r"/media/ubuntu/2C8A93A48A93695A/ubuntu/Week 45/student_performance/data/student_scores_updated.csv")

    model, features_test, target_test = train_model(df)

    mse = evaluate_model(model, features_test, target_test)
    print(f"Model MSE: {mse}")

    joblib.dump(model, "model.pkl")


if __name__ == "__main__":
    main()


