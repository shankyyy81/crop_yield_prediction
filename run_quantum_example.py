#!/usr/bin/env python3
"""
Simple example script to run quantum crop yield prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.algorithms import VQR
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic crop yield data
    rainfall = np.random.uniform(500, 2000, n_samples)
    temperature = np.random.uniform(15, 35, n_samples)
    pesticides = np.random.uniform(10000, 80000, n_samples)
    
    # Create target variable with some noise
    yield_target = (0.1 * rainfall + 0.05 * temperature + 0.001 * pesticides + 
                   np.random.normal(0, 1000, n_samples))
    
    data = pd.DataFrame({
        'Rainfall (mm)': rainfall,
        'Temperature (Celsius)': temperature,
        'Pesticides (tonnes)': pesticides,
        'Yield (hg/ha)': yield_target
    })
    
    return data

def run_quantum_experiment(data):
    """Run quantum experiment with the provided data"""
    
    print("=== Quantum Crop Yield Prediction ===")
    print(f"Dataset shape: {data.shape}")
    
    # Prepare data
    features = ['Rainfall (mm)', 'Temperature (Celsius)', 'Pesticides (tonnes)']
    target = 'Yield (hg/ha)'
    
    X = data[features]
    y = data[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Setup quantum backend (simulator for this example)
    backend = QasmSimulator()
    quantum_instance = QuantumInstance(backend, shots=1024)
    
    # Create quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
    
    # Create quantum ansatz
    ansatz = RealAmplitudes(X_train.shape[1], reps=2)
    
    # Create VQR model
    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=50),  # Reduced for faster execution
        quantum_instance=quantum_instance
    )
    
    print("\nTraining quantum model...")
    vqr.fit(X_train, y_train)
    print("Training completed!")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = vqr.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n=== Results ===")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield (hg/ha)')
    plt.ylabel('Predicted Yield (hg/ha)')
    plt.title('Quantum VQR Model Predictions')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def run_on_ibm_quantum_demo():
    """Demonstrate how to run on IBM quantum hardware"""
    print("\n=== IBM Quantum Hardware Demo ===")
    print("To run on real IBM quantum hardware:")
    print("1. Get IBM Quantum account: https://quantum-computing.ibm.com/")
    print("2. Get API token from your account")
    print("3. Install qiskit-ibmq-provider: pip install qiskit-ibmq-provider")
    print("4. Use the following code:")
    
    demo_code = '''
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# Load your account
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

# Find available backends
available_backends = provider.backends(
    filters=lambda x: x.configuration().n_qubits >= 4 
    and not x.configuration().simulator
)

# Use least busy backend
backend = least_busy(available_backends)
print(f"Using backend: {backend.name()}")

# Create quantum instance with real backend
quantum_instance = QuantumInstance(backend, shots=1024)

# Run your quantum model with real backend
# (Same code as above, but with real quantum_instance)
'''
    
    print(demo_code)

def main():
    """Main function to run the example"""
    
    # Create sample data
    print("Creating sample data...")
    data = create_sample_data()
    
    # Run quantum experiment
    results = run_quantum_experiment(data)
    
    # Show IBM quantum demo
    run_on_ibm_quantum_demo()
    
    print("\n=== Summary ===")
    print("✅ Quantum model trained successfully!")
    print("✅ Predictions made and evaluated!")
    print("✅ Results plotted!")
    print("\nNext steps:")
    print("1. Try with your real crop yield data")
    print("2. Experiment with different quantum circuits")
    print("3. Run on real IBM quantum hardware")
    print("4. Compare with classical models")

if __name__ == "__main__":
    main() 