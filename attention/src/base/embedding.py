import torch
import torch.nn as nn


class MyEmbedding(nn.Embedding):
    """Embedding base class."""

    def __init__(self, vocab_size, embedding_size, update_embedding=False, reduction='none', use_tfidf_weights=False,
                 normalize=False):
        super().__init__(vocab_size, embedding_size)
        print('embedding.py: Using word embeddings!')

        # Check if choice of reduction is valid
        assert reduction in ('none', 'mean', 'max')

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight.requires_grad = update_embedding
        self.reduction = reduction
        self.use_tfidf_weights = use_tfidf_weights
        self.normalize = normalize

    def forward(self, x, weights=None, clf_tokens_mask=None, padding_mask = None):
        # x.shape = (sentence_length, batch_size)
        # weights.shape = (sentence_length, batch_size)

        embedded = super().forward(x)
        # embedded.shape = (sentence_length, batch_size, embedding_size)

        # Reduce representation if specified to (weighted) mean of document word vector embeddings over sentence_length
        #   'mean' : (weighted) mean of document word vector embeddings over sentence_length
        #   'max'  : max-pooling of document word vector embedding dimensions over sentence_length
        # After reduction: embedded.shape = (batch_size, embedding_size)
        if self.reduction != 'none':

            if self.reduction == 'mean':
                if self.use_tfidf_weights:
                    # compute tf-idf weighted mean if specified
                    embedded = torch.sum(embedded * weights.unsqueeze(2), dim=0)
                else:
                    embedded = torch.mean(embedded, dim=0)

            if self.reduction == 'max':
                embedded, _ = torch.max(embedded, dim=0)

            if self.normalize:
                embedded = embedded / torch.norm(embedded, p=2, dim=1, keepdim=True).clamp(min=1e-08)
                embedded[torch.isnan(embedded)] = 0

        return embedded

    
    


class MyEmbeddingFrequency(nn.Embedding):
    """Embedding base class."""

    def __init__(self, vocab_size):
        super().__init__(vocab_size, vocab_size)
        print('embedding.py: Using word Frequency!')
        # Check if choice of reduction is valid

        self.vocab_size = vocab_size
        self.embedding_size = vocab_size


    def forward(self, x, weights=None, clf_tokens_mask=None, padding_mask = None):
        # x.shape = (sentence_length, batch_size)
        # weights.shape = (sentence_length, batch_size)

        bs = x.shape[1]
        embedded = torch.zeros((x.shape[0], bs, self.vocab_size))
        for i in range(bs):
            for j,idx in enumerate(x[:,i]):
                embedded[j,i,idx]+=1.
                
        # hidden.shape = (sentence_length, batch_size, hidden_size)


        return embedded.cuda()
