import matplotlib.pyplot as plt

def plot_results(test_df, y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, y_test.values, label="Actual", marker='o')
    plt.plot(test_df.index, y_pred, label="Predicted", marker='x')
    plt.xlabel("Index")
    plt.ylabel("PM2.5 Levels")
    plt.title("Actual vs Predicted PM2.5")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()