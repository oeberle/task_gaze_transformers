from ..base.embedding import MyEmbedding, MyEmbeddingFrequency
from ..utils.word_vectors import load_word_vectors


def build_embedding(dataset, embedding_size=None, pretrained_model=None, update_embedding=True,
                  embedding_reduction='none', use_tfidf_weights=False, normalize_embedding=False,
                  word_vectors_cache='../data/word_vectors_cache'):
    """Builds the neural network."""

    vocab_size = dataset.encoder.vocab_size
    
    # Set embedding

    # Load pre-trained model if specified
    embedding = None
    if pretrained_model is not None:
        # if word vector model
        if pretrained_model in ['GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en']:
            word_vectors, embedding_size = load_word_vectors(pretrained_model, embedding_size, word_vectors_cache)
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
            # Init embedding with pre-trained word vectors
            for i, token in enumerate(dataset.encoder.tokenizer.vocab):
                embedding.weight.data[i] = word_vectors[token]
        # if language model
       # elif pretrained_model in ['bert']:
       #     embedding = BERT()
            
       # elif pretrained_model in ['transformer']:
       #     embedding = Transformer()
        elif pretrained_model in ['termfreq']:
            embedding = MyEmbeddingFrequency(vocab_size)
             
            
    else:
        if embedding_size is not None:
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
        else:
            raise Exception('If pretrained_model is None, embedding_size must be specified')
            
    return embedding
