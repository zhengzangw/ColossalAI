import torch
import transformers

from ..registry import ModelAttribute, model_zoo

# ===============================
# Register single-sentence GPT
# ===============================


def data_gen():
    # Generated from following code snippet
    #
    # from transformers import GPT2Tokenizer
    # input = 'Hello, my dog is cute'
    # tokenized_input = tokenizer(input, return_tensors='pt')
    # input_ids = tokenized_input['input_ids']
    # attention_mask = tokenized_input['attention_mask']
    input_ids = torch.tensor([[15496, 11, 616, 3290, 318, 13779]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.int64)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def data_gen_for_lm():
    # LM data gen
    # the `labels` of LM is the token of the output, cause no padding, use `input_ids` as `labels`
    data = data_gen()
    data['labels'] = data['input_ids'].clone()
    return data


def data_gen_for_token_classification():
    # token classification data gen
    # `labels` is the type not the token id for token classification, 0 or 1
    data = data_gen()
    data['labels'] = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int64)
    return data


def data_gen_for_sequence_classification():
    # sequence classification data gen
    data = data_gen()
    data['labels'] = torch.tensor([0], dtype=torch.int64)
    return data


# define output transform function
output_transform_fn = lambda x: x

# define loss function
loss_fn_for_gpt2_model = lambda x: x.last_hidden_state.mean()
loss_fn = lambda x: x.loss

config = transformers.GPT2Config(n_layer=2,
                                 n_head=4,
                                 vocab_size=50258,
                                 attn_pdrop=0,
                                 embd_pdrop=0,
                                 resid_pdrop=0,
                                 summary_first_dropout=0,
                                 hidden_dropout=0,
                                 problem_type="single_label_classification")

# register the following models
model_zoo.register(name='transformers_gpt',
                   model_fn=lambda: transformers.GPT2Model(config),
                   data_gen_fn=data_gen,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn_for_gpt2_model,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_lm',
                   model_fn=lambda: transformers.GPT2LMHeadModel(config),
                   data_gen_fn=data_gen_for_lm,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_double_heads',
                   model_fn=lambda: transformers.GPT2DoubleHeadsModel(config),
                   data_gen_fn=data_gen_for_lm,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_for_token_classification',
                   model_fn=lambda: transformers.GPT2ForTokenClassification(config),
                   data_gen_fn=data_gen_for_token_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
model_zoo.register(name='transformers_gpt_for_sequence_classification',
                   model_fn=lambda: transformers.GPT2ForSequenceClassification(config),
                   data_gen_fn=data_gen_for_sequence_classification,
                   output_transform_fn=output_transform_fn,
                   loss_fn=loss_fn,
                   model_attribute=ModelAttribute(has_control_flow=True))
