class HookedEncoder(nn.Module):
    def __init__(self):
        super(HookedEncoder, self).__init__()

        self.attention_weights = []

        self.encoder_layer = model.encoder.encoder_layer
        self.transformer_encoder = model.encoder.transformer_encoder

        self.norm1 = model.encoder.norm1
        self.norm2 = model.encoder.norm2
        self.feed_forward = model.encoder.feed_forward

        def hook_fn(module, input, output):
            attn_weights = output[1]
            self.attention_weights.append(attn_weights)

        for layer in self.transformer_encoder.layers:
            layer.self_attn.register_forward_hook(hook_fn)

    def forward(self, x):
        residual = x

        x_norm = self.norm1(x)
        x_trans = self.transformer_encoder(x_norm)

        x_norm_2 = self.norm2(x_trans)
        ff_out = self.feed_forward(x_norm_2)

        return ff_out + residual

class Truncated_Model(nn.Module):
    def __init__(self):
        super(Truncated_Model, self).__init__()

        self.linear1 = model.linear1
        self.layer_norm1 = model.layer_norm1
        self.activation1 = model.activation1
        self.encoder = HookedEncoder()

    def forward(self, x):
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = self.activation1(x)
        encoded_states = self.encoder(x)

        return encoded_states, self.encoder.attention_weights


trunc_model = Truncated_Model().to(device)