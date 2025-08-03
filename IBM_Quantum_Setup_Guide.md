# IBM Quantum Setup Guide for Crop Yield Prediction

This guide will help you run your crop yield prediction model on IBM quantum computers using Qiskit.

## Prerequisites

### 1. IBM Quantum Account Setup

1. **Create IBM Quantum Account**:
   - Go to [IBM Quantum Experience](https://quantum-computing.ibm.com/)
   - Sign up for a free account
   - Verify your email address

2. **Get API Token**:
   - Log into IBM Quantum Experience
   - Go to "Account" → "API token"
   - Click "Create token"
   - Copy and save your token securely

### 2. Install Dependencies

```bash
# Install quantum dependencies
pip install -r requirements_quantum.txt

# Or install individually
pip install qiskit qiskit-machine-learning qiskit-aer qiskit-ibmq-provider
```

### 3. Configure IBM Quantum Access

Create a Python script to configure your IBM Quantum access:

```python
from qiskit import IBMQ

# Save your API token
IBMQ.save_account('YOUR_API_TOKEN_HERE')

# Load your account
IBMQ.load_account()
```

## Running Your Model on IBM Quantum

### Option 1: Using the Quantum Script

```python
from quantum_crop_yield_prediction import QuantumCropYieldPredictor, run_on_ibm_quantum

# Method 1: Run on simulator first (recommended for testing)
predictor = QuantumCropYieldPredictor(n_qubits=4, n_layers=2)
metrics, predictions = predictor.run_quantum_experiment('your_data.csv')

# Method 2: Run on real IBM quantum hardware
results = run_on_ibm_quantum()
```

### Option 2: Step-by-Step Process

```python
from quantum_crop_yield_prediction import QuantumCropYieldPredictor
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

# 1. Load IBM Quantum account
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')

# 2. Find available backends
available_backends = provider.backends(
    filters=lambda x: x.configuration().n_qubits >= 4 
    and not x.configuration().simulator
)

# 3. Select least busy backend
backend = least_busy(available_backends)
print(f"Using backend: {backend.name()}")

# 4. Create quantum predictor
predictor = QuantumCropYieldPredictor(n_qubits=4, n_layers=2)

# 5. Set up quantum instance with real backend
from qiskit.utils import QuantumInstance
predictor.quantum_instance = QuantumInstance(backend, shots=1024)

# 6. Run experiment
metrics, predictions = predictor.run_quantum_experiment('your_data.csv')
```

## Available IBM Quantum Backends

### Free Tier Backends
- **ibmq_manila**: 5 qubits
- **ibmq_belem**: 5 qubits  
- **ibmq_lima**: 5 qubits
- **ibmq_nairobi**: 7 qubits

### Premium Backends (requires subscription)
- **ibmq_bogota**: 5 qubits
- **ibmq_santiago**: 5 qubits
- **ibmq_athens**: 5 qubits
- **ibmq_rome**: 5 qubits

## Quantum Circuit Architecture

The quantum model uses:

1. **Feature Map**: ZZFeatureMap to encode classical data into quantum states
2. **Ansatz**: RealAmplitudes for parameterized quantum circuit
3. **Optimizer**: SPSA (Simultaneous Perturbation Stochastic Approximation)
4. **Algorithm**: VQR (Variational Quantum Regressor)

## Performance Considerations

### Simulator vs Real Hardware

| Aspect | Simulator | Real Hardware |
|--------|-----------|---------------|
| Speed | Fast | Slower (queue time) |
| Accuracy | Perfect | Noisy (decoherence) |
| Cost | Free | Free tier limits |
| Availability | Always | Limited access |

### Optimization Tips

1. **Start with Simulator**: Always test on simulator first
2. **Use Small Circuits**: Start with 2-4 qubits
3. **Reduce Shots**: Use 1024 shots instead of 8192 for faster results
4. **Choose Least Busy Backend**: Reduces queue time
5. **Monitor Job Status**: Real hardware jobs can take time

## Error Handling

```python
try:
    results = run_on_ibm_quantum()
except Exception as e:
    print(f"Error: {e}")
    print("Falling back to simulator...")
    predictor = QuantumCropYieldPredictor()
    results = predictor.run_quantum_experiment('your_data.csv')
```

## Comparing Classical vs Quantum

```python
from quantum_crop_yield_prediction import compare_classical_vs_quantum

# Compare performance
classical_results, quantum_metrics = compare_classical_vs_quantum('your_data.csv')
```

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   ```python
   # Re-save your token
   from qiskit import IBMQ
   IBMQ.save_account('YOUR_NEW_TOKEN')
   ```

2. **No Backends Available**:
   ```python
   # Check available backends
   from qiskit import IBMQ
   provider = IBMQ.get_provider()
   print(provider.backends())
   ```

3. **Job Queue Time**:
   ```python
   # Check job status
   job = backend.run(circuit)
   print(f"Job ID: {job.job_id()}")
   print(f"Status: {job.status()}")
   ```

4. **Memory Issues**:
   - Reduce number of qubits
   - Use smaller datasets
   - Increase shots gradually

### Getting Help

- [IBM Quantum Documentation](https://qiskit.org/documentation/)
- [Qiskit Machine Learning Guide](https://qiskit.org/documentation/machine-learning/)
- [IBM Quantum Community](https://quantum-computing.ibm.com/community)

## Example Output

```
Setting up quantum backend...
Using IBM Quantum backend: ibmq_manila
Preparing data...
Data shape: X_train (194, 3), X_test (49, 3)
Creating quantum circuit...
Quantum circuit created with 3 qubits
Creating quantum model...
Training quantum model...
Training completed!
Making predictions...
Evaluating model...
Mean Squared Error: 45678.23
Root Mean Squared Error: 213.72
Mean Absolute Error: 156.89
R² Score: 0.9234
```

## Next Steps

1. **Experiment with Different Circuits**: Try different feature maps and ansatzes
2. **Hyperparameter Tuning**: Adjust number of layers and qubits
3. **Hybrid Models**: Combine quantum and classical approaches
4. **Real-time Monitoring**: Set up job monitoring for long-running experiments
5. **Cost Optimization**: Monitor your IBM Quantum usage

## Cost Considerations

- **Free Tier**: 1000 shots per month
- **Premium**: Additional shots and priority access
- **Monitor Usage**: Check your IBM Quantum dashboard regularly

Remember: Quantum computing is still in development. Results may vary and classical methods might perform better for some datasets. 