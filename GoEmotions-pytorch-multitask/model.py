import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, pooled_output, (hidden_states), (attentions)


class SubtaskModule(nn.Module):
    def __init__(self, hidden_size, subtask_num_labels, hidden_dropout_prob):
        super(SubtaskModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = subtask_num_labels
        self.classifier = nn.Linear(hidden_size, subtask_num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        nn.init.kaiming_normal(self.classifier.weight)

    def forward(self, pooled_output, labels):
        out = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fct(logits, labels)
        return loss
