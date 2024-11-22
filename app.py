"""
Hyperbolic Embeddings Framework
MIT License

Copyright (c) 2024 Richard Aragon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import geoopt


class HyperbolicEmbedding(torch.nn.Module):
    """
    Hyperbolic Embedding Layer using Poincaré Ball.
    """
    def __init__(self, num_embeddings, embedding_dim, curvature=1.0):
        super(HyperbolicEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.manifold = geoopt.PoincareBall(c=curvature)
        self.embeddings = geoopt.ManifoldParameter(
            self.manifold.random_normal((num_embeddings, embedding_dim), std=1e-2),
            manifold=self.manifold
        )

    def forward(self, indices):
        return self.embeddings[indices]


class HyperbolicPositionalEncoding(torch.nn.Module):
    """
    Hyperbolic Positional Encoding for token sequences.
    """
    def __init__(self, max_len, embedding_dim, curvature=1.0):
        super(HyperbolicPositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.manifold = geoopt.PoincareBall(c=curvature)
        self.positions = geoopt.ManifoldParameter(
            self.manifold.random_normal((max_len, embedding_dim), std=1e-2),
            manifold=self.manifold
        )

    def forward(self, seq_len, batch_size):
        pos_encodings = self.positions[:seq_len]
        pos_encodings = pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        return pos_encodings


class MultiHeadHyperbolicAttention(torch.nn.Module):
    """
    Multi-Head Attention Mechanism in Hyperbolic Space.
    """
    def __init__(self, embedding_dim, num_heads, curvature=1.0):
        super(MultiHeadHyperbolicAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.manifold = geoopt.PoincareBall(c=curvature)

        self.query_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.output_proj = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        dist_matrix = self.manifold.dist(Q.unsqueeze(3), K.unsqueeze(2))
        attention_weights = torch.exp(-dist_matrix)
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)

        output = torch.matmul(attention_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)

        return self.output_proj(output), attention_weights


class HyperbolicTransformerEncoder(torch.nn.Module):
    """
    Hyperbolic Transformer Encoder Block.
    """
    def __init__(self, num_embeddings, embedding_dim, num_heads, max_len, curvature=1.0):
        super(HyperbolicTransformerEncoder, self).__init__()
        self.embedding = HyperbolicEmbedding(num_embeddings, embedding_dim, curvature)
        self.positional_encoding = HyperbolicPositionalEncoding(max_len, embedding_dim, curvature)
        self.attention = MultiHeadHyperbolicAttention(embedding_dim, num_heads, curvature)
        self.fc = torch.nn.Linear(embedding_dim, embedding_dim)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, indices):
        batch_size, seq_len = indices.size()
        embeddings = self.embedding(indices.view(-1)).view(batch_size, seq_len, -1)
        pos_encodings = self.positional_encoding(seq_len, batch_size)
        embeddings = embeddings + pos_encodings

        attention_output, _ = self.attention(embeddings, embeddings, embeddings)
        transformed = self.fc(attention_output)
        return self.dropout(self.activation(transformed))
