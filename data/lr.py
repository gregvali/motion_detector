# Linear Regression vs Neural Network Comparison

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data(X, y):
    """Explore the data to understand its characteristics"""
    print("=== DATA EXPLORATION ===")
    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {np.mean(y):.4f}")
    print(f"  Std: {np.std(y):.4f}")
    print(f"  Min: {np.min(y):.4f}")
    print(f"  Max: {np.max(y):.4f}")
    
    # Check for correlations
    # if X.shape[1] <= 10:  # Only for small feature sets
    #     print(f"\nFeature correlations with target:")
    #     for i in range(X.shape[1]):
    #         corr = np.corrcoef(X[:, i], y)[0, 1]
    #         print(f"  Feature {i}: {corr:.4f}")
    
    return X, y

def compare_linear_models(X, y):
    """Compare different linear regression models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    print("\n=== LINEAR MODEL COMPARISON ===")
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'predictions': y_pred_test
        }
        
        print(f"\n{name}:")
        print(f"  Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
        print(f"  Train R²:  {train_r2:.4f} | Test R²:  {test_r2:.4f}")
        print(f"  Test MAE:  {test_mae:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results, X_test_scaled, y_test, scaler

def try_polynomial_features(X, y, degrees=[2, 3]):
    """Try polynomial features with linear regression"""
    print("\n=== POLYNOMIAL FEATURES ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    for degree in degrees:
        print(f"\nPolynomial degree {degree}:")
        
        # Create polynomial pipeline
        poly_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        
        # Train
        poly_model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = poly_model.predict(X_train)
        y_pred_test = poly_model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(poly_model, X_train, y_train, cv=5, scoring='r2')
        
        results[f'Poly_{degree}'] = {
            'model': poly_model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        print(f"  Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")
        print(f"  Train R²:  {train_r2:.4f} | Test R²:  {test_r2:.4f}")
        print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print(f"  ⚠️  Potential overfitting detected!")
    
    return results

def compare_with_neural_network(X, y, linear_results):
    """Compare best linear model with neural network"""
    print("\n=== NEURAL NETWORK COMPARISON ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple neural network
    nn_model = MLPRegressor(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42
    )
    
    nn_model.fit(X_train_scaled, y_train)
    
    y_pred_nn = nn_model.predict(X_test_scaled)
    nn_mse = mean_squared_error(y_test, y_pred_nn)
    nn_r2 = r2_score(y_test, y_pred_nn)
    
    print(f"Neural Network:")
    print(f"  Test MSE: {nn_mse:.4f}")
    print(f"  Test R²:  {nn_r2:.4f}")
    print(f"  Training iterations: {nn_model.n_iter_}")
    
    # Find best linear model
    best_linear = max(linear_results.items(), key=lambda x: x[1]['test_r2'])
    best_name, best_results = best_linear
    
    print(f"\nBest Linear Model: {best_name}")
    print(f"  Test MSE: {best_results['test_mse']:.4f}")
    print(f"  Test R²:  {best_results['test_r2']:.4f}")
    
    print(f"\nComparison:")
    if best_results['test_r2'] > nn_r2:
        print(f"🏆 Linear model wins! (R² difference: {best_results['test_r2'] - nn_r2:.4f})")
    else:
        print(f"🏆 Neural network wins! (R² difference: {nn_r2 - best_results['test_r2']:.4f})")
    
    return nn_model, best_linear

def plot_results(linear_results, y_test):
    """Plot actual vs predicted for different models"""
    n_models = len(linear_results)
    fig, axes = plt.subplots(1, min(n_models, 4), figsize=(15, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, results) in enumerate(list(linear_results.items())[:4]):
        if i < len(axes):
            axes[i].scatter(y_test, results['predictions'], alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'{name}\nR² = {results["test_r2"]:.4f}')
    
    plt.tight_layout()
    plt.show()

def diagnose_poor_performance(X, y, results):
    """Diagnose why the model might be performing poorly"""
    print("\n=== PERFORMANCE DIAGNOSIS ===")
    
    best_r2 = max(result['test_r2'] for result in results.values())
    
    if best_r2 < 0.3:
        print("🔍 Poor performance detected. Possible causes:")
        print("  1. Data might not have a strong linear relationship")
        print("  2. Important features might be missing")
        print("  3. Data might need more preprocessing")
        print("  4. Target variable might be very noisy")
        print("  5. Non-linear relationships might exist")
        
        # Check target distribution
        print(f"\nTarget variable analysis:")
        print(f"  Coefficient of variation: {np.std(y)/np.mean(y):.4f}")
        if np.std(y)/np.mean(y) > 1:
            print("  ⚠️  High variability in target - consider log transformation")
        
        # Suggest next steps
        print(f"\n💡 Suggestions:")
        print("  - Try feature engineering (interactions, polynomials)")
        print("  - Check for outliers and remove them")
        print("  - Consider log/sqrt transformation of target")
        print("  - Try ensemble methods (Random Forest, Gradient Boosting)")
        print("  - Collect more relevant features")

# ==================== EXAMPLE USAGE ====================

def run_comprehensive_analysis():
    """Run complete analysis with sample data"""
    print("COMPREHENSIVE REGRESSION ANALYSIS")
    print("=" * 50)


    df = pd.read_csv('event_df.csv')

    df['minutes_until_next'] = df['time_since_last'].shift(-1)
    df = df[:-1]

    features = ['hour', 'day_of_week', 'is_weekend', 'time_since_last',
           'events_last_hour', 'avg_interval_last_hour']

    X = df[features]
    y = df['minutes_until_next']
    
    # Explore data
    X, y = load_and_explore_data(X, y)
    
    # Compare linear models
    linear_results, X_test_scaled, y_test, scaler = compare_linear_models(X, y)
    
    # Try polynomial features
    poly_results = try_polynomial_features(X, y)
    
    # Compare with neural network
    nn_model, best_linear = compare_with_neural_network(X, y, linear_results)
    
    # Plot results
    plot_results(linear_results, y_test)
    
    # Diagnose performance
    diagnose_poor_performance(X, y, linear_results)
    
    return linear_results, poly_results, nn_model

if __name__ == "__main__":
    # For your own data, replace this with:
    # X = your_feature_data  # Shape: (n_samples, n_features)
    # y = your_target_data   # Shape: (n_samples,)
    
    results = run_comprehensive_analysis()