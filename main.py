from utils.data.load_data import load_data
from model.Model import Model


def main():
    # Load dataset
    x_train, y_train, x_test, y_test, prev_y, unnormalized, forecast = load_data(
        "./data/bitcoin_historical.csv", 50)

    # Calculate the input shape of the data
    input_shape = (forecast, x_train.shape[-1])

    # Create and build the model
    model = Model()
    model.build(forecast, 0.2, 'linear', 'mse', 'adam', input_shape)

    # Train the model and get the time taken
    model.train(x_train, y_train, 1024, 5, .05)
    model.train_time()

    # Make predictions
    y_predict, y_predict_actual, y_test_actual = model.test(
        x_test, y_test, unnormalized)

    # TODO: Hook up to AWS Sagemaker
    # TODO: Plot Predition
    # TODO: Plot Changes
    # TODO: Plot Model Statistics


if __name__ == "__main__":
    main()
