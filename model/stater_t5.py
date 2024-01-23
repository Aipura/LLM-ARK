from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Model
import numpy as np
import os
logger = logging.get_logger(__name__)

from transformers import T5PreTrainedModel
from collections import OrderedDict
_CONFIG_FOR_DOC = "T5Config"
from pprint import pprint

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        # self.dense = nn.Linear(config.d_model, config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.d_model)
        self.fc2 = nn.Linear(config.d_model, config.d_model)
        # self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.actor = nn.Linear(config.d_model, 400)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.actor(hidden_states)
        return hidden_states

class StaterT5(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, graph=None):
        super().__init__(config)
        # config.num_labels = 400,
        self.transformer = T5Model(config)
        self.classification_head = T5ClassificationHead(config)
        # self.fc1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.fc2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # self.activate_func = nn.Tanh()
        # self.actor = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.graph = graph

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        current_entities: torch.LongTensor = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        return sentence_representation
