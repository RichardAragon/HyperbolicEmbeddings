## **Hyperbolic Embeddings Framework**

### **Overview**
The **Hyperbolic Embeddings Framework** is a Python library designed to simplify the implementation of hyperbolic neural networks. It includes modular components like hyperbolic embeddings, positional encodings, and attention mechanisms, making it easy to build hyperbolic Transformers and other models that operate in curved spaces.

This framework is ideal for tasks involving hierarchical, relational, or tree-structured data, such as:
- Knowledge graph representation.
- Taxonomy classification.
- Natural language understanding with long-range dependencies.

---

### **Key Features**
- **Hyperbolic Embeddings**: Efficient token representations in Poincaré space.
- **Hyperbolic Positional Encoding**: Handles sequence-based data with respect to hyperbolic geometry.
- **Multi-Head Hyperbolic Attention**: Scales attention mechanisms to hyperbolic spaces.
- **Transformer Encoder**: Modular and extensible Transformer components for curved spaces.

---

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/RichardAragon/hyperbolic-embeddings.git
   cd hyperbolic-embeddings
   ```

2. Install the required Python packages:
   ```bash
   pip install torch geoopt
   ```

---

### **Quick Start**

Here’s an example of how to use the framework to create a simple hyperbolic Transformer encoder:

```python
import torch
from hyperbolic_embeddings import HyperbolicTransformerEncoder
from geoopt.optim import RiemannianAdam

# Define model parameters
num_embeddings = 10000
embedding_dim = 128
num_heads = 4
max_len = 50
curvature = 1.0

# Instantiate the hyperbolic Transformer encoder
model = HyperbolicTransformerEncoder(
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    max_len=max_len,
    curvature=curvature
)

# Example input: token indices
input_indices = torch.randint(0, num_embeddings, (32, max_len))  # Batch of 32 sequences

# Forward pass
output = model(input_indices)  # Output shape: [batch_size, seq_len, embedding_dim]
print(output.shape)

# Use RAdam optimizer
optimizer = RiemannianAdam(model.parameters(), lr=0.01)

# Example training loop
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    optimizer.zero_grad()
    logits = model(input_indices)
    loss = criterion(logits.mean(dim=1), torch.randint(0, 2, (32,)))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

### **Key Recommendation**
#### **Use Riemannian Adam (RAdam)**
From our testing, the **RAdam optimizer** consistently outperforms traditional optimizers like Adam and RSGD in hyperbolic spaces. It provides:
- Faster convergence.
- Stability in optimization.
- Superior performance on complex tasks.

To use RAdam, install it via `geoopt` and apply it as shown above.

---

### **Folder Structure**
```plaintext
hyperbolic-embeddings/
├── README.md               # Overview and usage instructions
├── LICENSE                 # MIT license
├── hyperbolic_embeddings.py  # Core framework code
└── examples/
    └── train_example.py    # Example training script
```

---

### **License**

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

### **Contributing**
We welcome contributions to the Hyperbolic Embeddings Framework! If you have ideas for improvements or additional features, feel free to:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Submit a pull request.

---

### **Future Directions**
- Add support for **hyperbolic decoding** and sequence-to-sequence models.
- Extend framework with **task-specific modules** for applications like graph representation and natural language inference.
- Benchmark the framework on large datasets to establish its versatility.


