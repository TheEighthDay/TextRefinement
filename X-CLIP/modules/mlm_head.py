

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

#https://github.com/huggingface/transformers/blob/f711d683b50e20f52778110e55e5a0894c2c8466/src/transformers/activations.py
#"gelu"

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = GELUActivation()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size,vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, hidden_size,vocab_size):
        super().__init__()
        self.predictions = BertLMPredictionHead(hidden_size,vocab_size)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


if __name__=="__main__":
    m = BertOnlyMLMHead(hidden_size=512,vocab_size=9000)
    a = torch.ones((2,32,512))
    b = m(a)
    print(b.shape)
    #(2,32,9000)


