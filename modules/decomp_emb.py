from torch import nn

class DecomposedEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        # Decomposed embedding: vocab_size x hidden_dim -> vocab_size x emb_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.projection = nn.Linear(hidden_dim, emb_dim, bias=False)
    
    def forward(self, x):
        # x shape: (*)
        embedded = self.embedding(x)  # shape: (*, hidden_dim)
        projected = self.projection(embedded)  # shape: (*, emb_dim)
        return projected