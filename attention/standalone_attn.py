
import torch
import torch.nn as nn

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

print(dc)

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)
print(sentence_int)


vocab_size = 50_000
torch.manual_seed(123)
embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)


torch.manual_seed(123)
d_in, d_out_kq, d_out_v = 3, 2, 4
W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
W_value = nn.Parameter(torch.rand(d_in, d_out_v))
x = embedded_sentence
keys = x @ W_key
queries = x @ W_query
values = x @ W_value
# attn_scores are the "omegas", 
# the unnormalized attention weights
attn_scores = queries @ keys.T 
print(attn_scores)
print(attn_scores.shape)