# RNN Models

### GRU

![GRU](./gru.png)

```python
# x[t: int] means time step, y[u: str] means param name.

u = sigmoid(x[t] * U['u']) + s[t-1] * W['u'])       # Update
r = sigmoid(x[t] * U['r']) + s[t-1] * W['r'])       # Reset
h = tanh(x[t] * U['h'] + dot(s[t-1], r) * W['h'])
s[t] = dot(1 - u, h) + dot(u, s[t-1])               # New hidden state
```


### LSTM

![LSTM](./lstm.png)
