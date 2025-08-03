#!/usr/bin/env python3
"""
Quantum Crop Yield Prediction using IBM Qiskit
This script implements quantum machine learning algorithms for crop yield prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit imports
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit_machine_learning.algorithms import VQR, VQC
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms.optimizers import ADAM

class QuantumCropYieldPredictor:
    """
    Quantum Machine Learning model for crop yield prediction
    """
    
    def __init__(self, n_qubits=4, n_layers=2, backend_type='qasm_simulator'):
        """
        Initialize the quantum predictor
        
        Args:
            n_qubits: Number of qubits to use
            n_layers: Number of layers in the quantum circuit
            backend_type: Type of quantum backend to use
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_type = backend_type
        self.scaler = StandardScaler()
        self.feature_map = None
        self.ansatz = None
        self.quantum_instance = None
        self.model = None
        
    def setup_quantum_backend(self):
        """Setup quantum backend"""
        if self.backend_type == 'qasm_simulator':
            backend = QasmSimulator()
            self.quantum_instance = QuantumInstance(backend, shots=1024)
        else:
            # For real quantum hardware, you would use IBMQ
            # from qiskit import IBMQ
            # IBMQ.load_account()
            # provider = IBMQ.get_provider(hub='ibm-q')
            # backend = provider.get_backend('ibmq_manila')  # or other available backend
            # self.quantum_instance = QuantumInstance(backend, shots=1024)
            raise NotImplementedError("Real quantum hardware setup not implemented yet")
    
    def create_quantum_circuit(self, n_features):
        """Create quantum feature map and ansatz"""
        # Feature map - encodes classical data into quantum state
        self.feature_map = ZZFeatureMap(feature_dimension=n_features, reps=1)
        
        # Ansatz - parameterized quantum circuit for learning
        self.ansatz = RealAmplitudes(n_features, reps=self.n_layers)
        
        # Combine feature map and ansatz
        qc = QuantumCircuit(n_features)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)
        
        return qc
    
    def prepare_data(self, data_path=None):
        """
        Prepare data for quantum model
        """
        if data_path:
            # Load data from file
            data = pd.read_csv(data_path)
        else:
            # Use the same data preparation as in the original notebook
            # This is a simplified version - you should adapt it to your actual data
            print("Please provide data_path or implement data loading logic")
            return None, None, None, None
        
        # Select numerical features for quantum model
        numerical_features = ['Rainfall (mm)', 'Temperature (Celsius)', 'Pesticides (tonnes)']
        target = 'Yield (hg/ha)'
        
        if all(feature in data.columns for feature in numerical_features + [target]):
            X = data[numerical_features]
            y = data[target]
            
            # Scale features to [0, 1] range for quantum encoding
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            return X_train, X_test, y_train, y_test
        else:
            print("Required features not found in data")
            return None, None, None, None
    
    def create_vqr_model(self):
        """Create Variational Quantum Regressor (VQR) model"""
        if self.feature_map is None or self.ansatz is None:
            raise ValueError("Feature map and ansatz must be created first")
        
        # Create VQR model
        self.model = VQR(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=100),
            quantum_instance=self.quantum_instance
        )
        
        return self.model
    
    def create_custom_quantum_model(self):
        """Create custom quantum neural network regressor"""
        if self.feature_map is None or self.ansatz is None:
            raise ValueError("Feature map and ansatz must be created first")
        
        # Create quantum neural network
        qnn = CircuitQNN(
            circuit=self.feature_map.compose(self.ansatz),
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            input_gradients=True,
            quantum_instance=self.quantum_instance
        )
        
        # Create regressor
        self.model = NeuralNetworkRegressor(
            neural_network=qnn,
            loss='squared_error',
            optimizer=ADAM(maxiter=100),
            initial_point=np.random.random(self.ansatz.num_parameters)
        )
        
        return self.model
    
    def train_model(self, X_train, y_train):
        """Train the quantum model"""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        print("Training quantum model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        return self.model
    
    def predict(self, X_test):
        """Make predictions using the trained quantum model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        return self.model.predict(X_test)
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def plot_results(self, y_true, y_pred, title="Quantum Model Predictions"):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Yield (hg/ha)')
        plt.ylabel('Predicted Yield (hg/ha)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def run_quantum_experiment(self, data_path=None):
        """Run complete quantum experiment"""
        print("Setting up quantum backend...")
        self.setup_quantum_backend()
        
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(data_path)
        
        if X_train is None:
            print("Data preparation failed")
            return
        
        print(f"Data shape: X_train {X_train.shape}, X_test {X_test.shape}")
        
        # Create quantum circuit
        print("Creating quantum circuit...")
        qc = self.create_quantum_circuit(X_train.shape[1])
        print(f"Quantum circuit created with {qc.num_qubits} qubits")
        
        # Create and train model
        print("Creating quantum model...")
        self.create_vqr_model()
        
        print("Training model...")
        self.train_model(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = self.predict(X_test)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = self.evaluate_model(y_test, y_pred)
        
        # Plot results
        self.plot_results(y_test, y_pred, "Quantum VQR Model Predictions")
        
        return metrics, y_pred

def run_on_ibm_quantum():
    """
    Function to run the model on IBM quantum hardware
    """
    from qiskit import IBMQ
    from qiskit.providers.ibmq import least_busy
    
    # Load IBM Quantum account
    try:
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        
        # Get available backends
        available_backends = provider.backends(
            filters=lambda x: x.configuration().n_qubits >= 4 
            and not x.configuration().simulator
        )
        
        if available_backends:
            # Use the least busy backend
            backend = least_busy(available_backends)
            print(f"Using IBM Quantum backend: {backend.name()}")
            
            # Create quantum predictor with real backend
            predictor = QuantumCropYieldPredictor(
                n_qubits=4, 
                n_layers=2, 
                backend_type='ibm_quantum'
            )
            
            # Set the real backend
            predictor.quantum_instance = QuantumInstance(backend, shots=1024)
            
            # Run experiment
            return predictor.run_quantum_experiment()
        else:
            print("No suitable IBM Quantum backends available")
            return None
            
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("Please ensure you have:")
        print("1. IBM Quantum account")
        print("2. API token configured")
        print("3. Internet connection")
        return None

def compare_classical_vs_quantum(data_path=None):
    """
    Compare classical and quantum model performance
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    # Prepare data
    predictor = QuantumCropYieldPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(data_path)
    
    if X_train is None:
        return
    
    # Classical models
    classical_models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    classical_results = {}
    
    print("Training classical models...")
    for name, model in classical_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = predictor.evaluate_model(y_test, y_pred)
        classical_results[name] = metrics
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    # Quantum model
    print("\nTraining quantum model...")
    predictor.setup_quantum_backend()
    predictor.create_quantum_circuit(X_train.shape[1])
    predictor.create_vqr_model()
    predictor.train_model(X_train, y_train)
    y_pred_quantum = predictor.predict(X_test)
    quantum_metrics = predictor.evaluate_model(y_test, y_pred_quantum)
    
    # Compare results
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for name, metrics in classical_results.items():
        print(f"\n{name}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
    
    print(f"\nQuantum VQR:")
    print(f"  R² Score: {quantum_metrics['r2']:.4f}")
    print(f"  RMSE: {quantum_metrics['rmse']:.2f}")
    
    return classical_results, quantum_metrics

if __name__ == "__main__":
    # Example usage
    print("Quantum Crop Yield Prediction")
    print("="*40)
    
    # Run with simulator (default)
    print("\n1. Running on Qiskit Aer Simulator...")
    predictor = QuantumCropYieldPredictor(n_qubits=4, n_layers=2)
    
    # You can provide your data file path here
    # metrics, predictions = predictor.run_quantum_experiment('your_data.csv')
    
    # For demonstration, we'll show the setup
    print("Quantum model setup completed!")
    print("To run on real IBM quantum hardware, use run_on_ibm_quantum()")
    print("To compare with classical models, use compare_classical_vs_quantum()") 