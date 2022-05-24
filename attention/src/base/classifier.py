import torch
import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, model, n_classes):
        super().__init__()
        self.model = model
        self.classification_head = nn.Linear(self.model.hidden_size, n_classes )
        
    def forward(self, inputs, clf_tokens_mask, clf_labels, padding_mask):
        if type(padding_mask) != type(None):
            padding_mask = padding_mask.transpose(1,0)
                                                     
        _, _, _, hidden_states = self.model(inputs, padding_mask)

        clf_logits = self.classification_head(hidden_states.squeeze(1))
        loss_fct = nn.CrossEntropyLoss()
      #  print(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))
        loss = loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))
        return clf_logits, loss
