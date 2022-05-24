import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads

        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)
        
    def forward(self, hidden):
        # hidden.shape = (sentence_length, batch_size, hidden_size)

        # Change to hidden.shape = (batch_size, sentence_length, hidden_size)
        
        x = torch.tanh(self.W1(hidden))
        # x.shape = (batch_size, sentence_length, attention_size)

        x = F.softmax(self.W2(x), dim=1)  # softmax over sentence_length
        # x.shape = (batch_size, sentence_length, n_attention_heads)

        A = x.transpose(1, 2)
        M = A @ hidden 
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)

        self.A = A
        return M

    
class SelfAttentionConv(nn.Module):

    def __init__(self, hidden_size, attention_size=100, n_attention_heads=1, compute_relevances=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.compute_relevances = compute_relevances

        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)
        

    def forward(self, hidden):
        # hidden.shape = (sentence_length, batch_size, hidden_size)
        
        x = torch.tanh(self.W1(hidden))
        # x.shape = (batch_size, sentence_length, attention_size)

        x = F.softmax(self.W2(x), dim=1)  # softmax over sentence_length
        # x.shape = (batch_size, sentence_length, n_attention_heads)

        A = x.transpose(1, 2)
        # A.shape = (batch_size, n_attention_heads, sentence_length)
        # M.shape = (batch_size, n_attention_heads, hidden_size)
      
        if self.compute_relevances == True:
            try:
                assert hidden.shape[0]==1., "Only bs=1 works for computing self-attention relevances"
            except:
                import pdb;pdb.set_trace()
               
            sen_len = hidden.shape[1]
            self.Attention = torch.nn.Conv2d(1, self.n_attention_heads, kernel_size = (1,sen_len),stride=(1,sen_len), bias=False, padding=(0,0))
            sbefore = self.Attention.weight.shape
            A_reshape = A.squeeze(0).unsqueeze(1).unsqueeze(1)
            self.Attention.weight = torch.nn.Parameter(A_reshape, requires_grad=True)
            assert sbefore == self.Attention.weight.shape, "{} {}".format(sbefore, self.Attention.weight.shape)
            hidden_flat = hidden.transpose(2,1).reshape(-1, self.hidden_size*sen_len)
            # Check for correct shape for convolution
            assert (hidden_flat[0,:sen_len] == hidden[0,:,0]).all()
            M = self.Attention(hidden_flat.unsqueeze(0).unsqueeze(0)).squeeze(2)

        else:
            M = A @ hidden 

        self.A = A
        return M
