import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_and_evaluate(csv_file):
    print("Loading data...")
    # Load the data
    data = pd.read_csv(csv_file)
    print("Data loaded successfully.")

    # EDA Visualizations
    print("Generating pairplot...")
    # Pairplot to visualize relationships between variables
    pairplot = sns.pairplot(data)
    pairplot.savefig("pairplot.png")
    print("Pairplot saved as 'pairplot.png'.")
    
    # Close the pairplot figure to prevent display
    plt.close()

    print("Generating correlation heatmap...")
    # Select numeric columns for correlation heatmap
    numeric_columns = data.select_dtypes(include=['number'])
    correlation_matrix = numeric_columns.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as 'correlation_heatmap.png'.")
    plt.close()

    # Select relevant columns
    relevant_columns = ['bet_qty_overall', 'bet_amount_overall', 'win_qty_overall', 'win_amount_overall']
    X = data[relevant_columns]
    y = data['ggr_overall']  # Assuming ggr_overall is the target variable (CLV)

    # Drop rows with missing values
    X = X.dropna()
    y = y[X.index]

    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split completed.")

    # Train the Linear Regression model
    print("Training Linear Regression model...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    print("Linear Regression model trained successfully.")

    # Train the Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")

    # Evaluate models
    print("Evaluating models...")
    linear_pred = linear_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    print("Models evaluated successfully.")

    linear_mse = mean_squared_error(y_test, linear_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)

    print("Linear Regression Mean Squared Error:", linear_mse)
    print("Random Forest Mean Squared Error:", rf_mse)

    # Print the coefficients of the linear model
    coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': linear_model.coef_})
    print("Linear Regression Coefficients:")
    print(coefficients)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_csv_file>")
        sys.exit(1)
    
    input_csv_file = sys.argv[1]
    train_and_evaluate(input_csv_file)
