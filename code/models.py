import torch
import torch.nn as nn
import numpy as np


class Model_mlp_mse(nn.Module):
    # NN with three relu hidden layers
    # quantile outputs are independent of eachother
    def __init__(
        self,
        n_input,
        n_hidden,
        n_output,
        is_dropout=False,
        is_batch=False,
        activation="relu",
    ):
        super(Model_mlp_mse, self).__init__()
        self.layer1 = nn.Linear(n_input, n_hidden, bias=True)
        self.layer2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer3 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.layer4 = nn.Linear(n_hidden, n_output, bias=True)
        self.drop1 = nn.Dropout(0.333)
        self.drop2 = nn.Dropout(0.333)
        self.drop3 = nn.Dropout(0.333)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.batch2 = nn.BatchNorm1d(n_hidden)
        self.batch3 = nn.BatchNorm1d(n_hidden)
        self.is_dropout = is_dropout
        self.is_batch = is_batch
        self.activation = activation
        self.loss_fn = nn.MSELoss()

    def forward_net(self, x):
        x = self.layer1(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        else:
            raise Exception("bad activation passed in")
        if self.is_dropout:
            x = self.drop1(x)
        if self.is_batch:
            x = self.batch1(x)

        x = self.layer2(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        if self.is_dropout:
            x = self.drop2(x)
        if self.is_batch:
            x = self.batch2(x)

        # uncomment for 3 hidden layer
        x = self.layer3(x)
        if self.activation == "relu":
            x = torch.relu(x)
        elif self.activation == "gelu":
            x = torch.nn.functional.gelu(x)
        if self.is_dropout:
            x = self.drop3(x)
        if self.is_batch:
            x = self.batch3(x)

        x = self.layer4(x)
        return x

    def forward(self, x):
        # we write this in this was so can reuse forward_net in Model_mlp_diff
        return self.forward_net(x)

    def loss_on_batch(self, x_batch, y_batch):
        # add this here so can sync w diffusion model
        y_pred_batch = self(x_batch)
        loss = self.loss_fn(y_pred_batch, y_batch)
        return loss

    def sample(self, x_batch):
        return self(x_batch)


class Model_mlp_diff(Model_mlp_mse):
    # this model just piggy backs onto the vanilla MLP
    # later on I'll use a fancier architecture, ie transformer
    # and also make it possible to condition on images
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        is_dropout=False,
        is_batch=False,
        activation="relu",
    ):
        n_input = x_dim + y_dim + 1
        n_output = y_dim
        super(Model_mlp_diff, self).__init__(n_input, n_hidden, n_output, is_dropout, is_batch, activation)

    def forward(self, y, x, t, context_mask):
        nn_input = torch.cat([y, x, t], dim=-1)
        return self.forward_net(nn_input)

    def loss_on_batch(self, x_batch, y_batch):
        # overwrite these methods as won't use them w diffusion model
        raise NotImplementedError

    def sample(self, x_batch):
        raise NotImplementedError


class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        # one layer of non-linearities (just a useful building block to use below)
        self.model = nn.Sequential(
            nn.Linear(in_feats, out_feats),
            nn.BatchNorm1d(num_features=out_feats),
            nn.GELU(),
        )

    def forward(self, x):
        return self.model(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, trans_emb_dim, transformer_dim, nheads):
        super(TransformerEncoderBlock, self).__init__()
        # mainly going off of https://jalammar.github.io/illustrated-transformer/

        self.trans_emb_dim = trans_emb_dim
        self.transformer_dim = transformer_dim
        self.nheads = nheads

        self.input_to_qkv1 = nn.Linear(self.trans_emb_dim, self.transformer_dim * 3)
        self.multihead_attn1 = nn.MultiheadAttention(self.transformer_dim, num_heads=self.nheads)
        self.attn1_to_fcn = nn.Linear(self.transformer_dim, self.trans_emb_dim)
        self.attn1_fcn = nn.Sequential(
            nn.Linear(self.trans_emb_dim, self.trans_emb_dim * 4),
            nn.GELU(),
            nn.Linear(self.trans_emb_dim * 4, self.trans_emb_dim),
        )
        self.norm1a = nn.BatchNorm1d(self.trans_emb_dim)
        self.norm1b = nn.BatchNorm1d(self.trans_emb_dim)

    def split_qkv(self, qkv):
        assert qkv.shape[-1] == self.transformer_dim * 3
        q = qkv[:, :, :self.transformer_dim]
        k = qkv[:, :, self.transformer_dim: 2 * self.transformer_dim]
        v = qkv[:, :, 2 * self.transformer_dim:]
        return (q, k, v)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        qkvs1 = self.input_to_qkv1(inputs)
        # shape out = [3, batchsize, transformer_dim*3]

        qs1, ks1, vs1 = self.split_qkv(qkvs1)
        # shape out = [3, batchsize, transformer_dim]

        attn1_a = self.multihead_attn1(qs1, ks1, vs1, need_weights=False)
        attn1_a = attn1_a[0]
        # shape out = [3, batchsize, transformer_dim = trans_emb_dim x nheads]

        attn1_b = self.attn1_to_fcn(attn1_a)
        attn1_b = attn1_b / 1.414 + inputs / 1.414  # add residual
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        attn1_b = self.norm1a(attn1_b.transpose(0, 2).transpose(0, 1))
        attn1_b = attn1_b.transpose(0, 1).transpose(0, 2)
        # batchnorm likes shape = [batchsize, trans_emb_dim, 3]
        # so have to shape like this, then return

        # fully connected layer
        attn1_c = self.attn1_fcn(attn1_b) / 1.414 + attn1_b / 1.414
        # shape out = [3, batchsize, trans_emb_dim]

        # normalise
        # attn1_c = self.norm1b(attn1_c)
        attn1_c = self.norm1b(attn1_c.transpose(0, 2).transpose(0, 1))
        attn1_c = attn1_c.transpose(0, 1).transpose(0, 2)
        return attn1_c


class Model_mlp_diff_embed(nn.Module):
    # this model embeds x, y, t, before input into a fc NN (w residuals)
    def __init__(
        self,
        x_dim,
        n_hidden,
        y_dim,
        embed_dim,
        output_dim=None,
        is_dropout=False,
        is_batch=False,
        activation="relu",
        net_type="fc",
        use_prev=False,
    ):
        super(Model_mlp_diff_embed, self).__init__()
        self.embed_dim = embed_dim  # input embedding dimension
        self.n_hidden = n_hidden
        self.net_type = net_type
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.use_prev = use_prev  # whether x contains previous timestep
        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # embedding NNs
        if self.use_prev:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(int(x_dim / 2), self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.x_embed_nn = nn.Sequential(
                nn.Linear(x_dim, self.embed_dim),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )  # no prev hist
        self.y_embed_nn = nn.Sequential(
            nn.Linear(y_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.t_embed_nn = TimeSiren(1, self.embed_dim)

        # fc nn layers
        if self.net_type == "fc":
            if self.use_prev:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 4, n_hidden))  # concat x, x_prev,
            else:
                self.fc1 = nn.Sequential(FCBlock(self.embed_dim * 3, n_hidden))  # no prev hist
            self.fc2 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))  # will concat y and t at each layer
            self.fc3 = nn.Sequential(FCBlock(n_hidden + y_dim + 1, n_hidden))
            self.fc4 = nn.Sequential(nn.Linear(n_hidden + y_dim + 1, self.output_dim))

        # transformer layers
        elif self.net_type == "transformer":
            self.nheads = 16  # 16
            self.trans_emb_dim = 64
            self.transformer_dim = self.trans_emb_dim * self.nheads  # embedding dim for each of q,k and v (though only k and v have to be same I think)

            self.t_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.y_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)
            self.x_to_input = nn.Linear(self.embed_dim, self.trans_emb_dim)

            self.pos_embed = TimeSiren(1, self.trans_emb_dim)

            self.transformer_block1 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block2 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block3 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)
            self.transformer_block4 = TransformerEncoderBlock(self.trans_emb_dim, self.transformer_dim, self.nheads)

            if self.use_prev:
                self.final = nn.Linear(self.trans_emb_dim * 4, self.output_dim)  # final layer params
            else:
                self.final = nn.Linear(self.trans_emb_dim * 3, self.output_dim)
        else:
            raise NotImplementedError

    def forward(self, y, x, t, context_mask):
        # embed y, x, t
        if self.use_prev:
            x_e = self.x_embed_nn(x[:, :int(self.x_dim / 2)])
            x_e_prev = self.x_embed_nn(x[:, int(self.x_dim / 2):])
        else:
            x_e = self.x_embed_nn(x)  # no prev hist
            x_e_prev = None
        y_e = self.y_embed_nn(y)
        t_e = self.t_embed_nn(t)

        # mask out context embedding, x_e, if context_mask == 1
        context_mask = context_mask.repeat(x_e.shape[1], 1).T
        x_e = x_e * (-1 * (1 - context_mask))
        if self.use_prev:
            x_e_prev = x_e_prev * (-1 * (1 - context_mask))

        # pass through fc nn
        if self.net_type == "fc":
            net_output = self.forward_fcnn(x_e, x_e_prev, y_e, t_e, x, y, t)

        # or pass through transformer encoder
        elif self.net_type == "transformer":
            net_output = self.forward_transformer(x_e, x_e_prev, y_e, t_e, x, y, t)

        return net_output

    def forward_fcnn(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        if self.use_prev:
            net_input = torch.cat((x_e, x_e_prev, y_e, t_e), 1)
        else:
            net_input = torch.cat((x_e, y_e, t_e), 1)
        nn1 = self.fc1(net_input)
        nn2 = self.fc2(torch.cat((nn1 / 1.414, y, t), 1)) + nn1 / 1.414  # residual and concat inputs again
        nn3 = self.fc3(torch.cat((nn2 / 1.414, y, t), 1)) + nn2 / 1.414
        net_output = self.fc4(torch.cat((nn3, y, t), 1))
        return net_output

    def forward_transformer(self, x_e, x_e_prev, y_e, t_e, x, y, t):
        # roughly following this: https://jalammar.github.io/illustrated-transformer/

        t_input = self.t_to_input(t_e)
        y_input = self.y_to_input(y_e)
        x_input = self.x_to_input(x_e)
        if self.use_prev:
            x_input_prev = self.x_to_input(x_e_prev)
        # shape out = [batchsize, trans_emb_dim]

        # add 'positional' encoding
        # note, here position refers to order tokens are fed into transformer
        t_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 1.0)
        y_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 2.0)
        x_input += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 3.0)
        if self.use_prev:
            x_input_prev += self.pos_embed(torch.zeros(x.shape[0], 1).to(x.device) + 4.0)

        if self.use_prev:
            inputs1 = torch.cat(
                (
                    t_input[None, :, :],
                    y_input[None, :, :],
                    x_input[None, :, :],
                    x_input_prev[None, :, :],
                ),
                0,
            )
        else:
            inputs1 = torch.cat((t_input[None, :, :], y_input[None, :, :], x_input[None, :, :]), 0)
        # shape out = [3, batchsize, trans_emb_dim]

        block1 = self.transformer_block1(inputs1)
        block2 = self.transformer_block2(block1)
        block3 = self.transformer_block3(block2)
        block4 = self.transformer_block4(block3)

        # flatten and add final linear layer
        # transformer_out = block2
        transformer_out = block4
        transformer_out = transformer_out.transpose(0, 1)  # roll batch to first dim
        # shape out = [batchsize, 3, trans_emb_dim]

        flat = torch.flatten(transformer_out, start_dim=1, end_dim=2)
        # shape out = [batchsize, 3 x trans_emb_dim]

        out = self.final(flat)
        # shape out = [batchsize, n_dim]
        return out


def ddpm_schedules(beta1, beta2, T, is_linear=True):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta1) * torch.arange(-1, T + 1, dtype=torch.float32) / T + beta1
    if is_linear:
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    else:
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class Model_Cond_Diffusion(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, x_dim, y_dim, drop_prob=0.1, guide_w=0.0):
        super(Model_Cond_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.nn_model = nn_model
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.guide_w = guide_w

    def loss_on_batch(self, x_batch, y_batch):
        _ts = torch.randint(1, self.n_T + 1, (y_batch.shape[0], 1)).to(self.device)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise = torch.randn_like(y_batch).to(self.device)

        # add noise to clean target actions
        y_t = self.sqrtab[_ts] * y_batch + self.sqrtmab[_ts] * noise

        # use nn model to predict noise
        noise_pred_batch = self.nn_model(y_t, x_batch, _ts / self.n_T, context_mask)

        # return mse between predicted and true noise
        return self.loss_mse(noise, noise_pred_batch)

    def sample(self, x_batch, return_y_trace=False, extract_embedding=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if extract_embedding:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask, x_embed)
            else:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T

        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # I'm a bit confused why we are adding noise during denoising?
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, extra_steps=4, return_y_trace=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(x_batch[:,0]).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):
            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i


# below stuff for outter MSE class
class Model_Cond_MSE(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim):
        super(Model_Cond_MSE, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_mse = nn.MSELoss()

    def loss_on_batch(self, x_batch, y_batch):
        # to keep mse as close as poss to the diffusion pipeline
        # we manufacture timesteps and context masks and noisy y
        # as we did for diffusion models
        # but these are just sent into the archicture as zeros for mse model
        # this means we can use _exactly_ the same architecture as for the diff model
        # although, I don't know if there's any point as we'll have to develop a new model later
        # for discretised model
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        return self.loss_mse(y_batch, y_pred)

    def sample(self, x_batch):
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        return self.nn_model(y_i, x_batch, _ts, context_mask)


def matrix_diag(diagonal):
    # batched diag operation
    # taken from here
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class Model_Cond_MeanVariance(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim):
        super(Model_Cond_MeanVariance, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim

    def loss_on_batch(self, x_batch, y_batch):
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)

        y_pred_mean = y_pred[:, :self.y_dim]
        y_pred_var = torch.log(1 + torch.exp(y_pred[:, self.y_dim:]))  # softplus ensure var>0

        covariance_matrix = matrix_diag(y_pred_var)
        # covariance_matrix is shape: batch_size, y_dim, y_dim, but off diagonal entries are zero
        dist = torch.distributions.multivariate_normal.MultivariateNormal(y_pred_mean, covariance_matrix)
        return -torch.mean(dist.log_prob(y_batch))  # return average negative log likelihood

    def sample(self, x_batch):
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)
        y_pred_mean = y_pred[:, :self.y_dim]
        y_pred_var = torch.log(1 + torch.exp(y_pred[:, self.y_dim:]))  # softplus ensure var>0

        covariance_matrix = matrix_diag(y_pred_var)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(y_pred_mean, covariance_matrix)
        return dist.sample()


class Model_Cond_Discrete(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, n_bins):
        super(Model_Cond_Discrete, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.n_bins = n_bins  # this is number of bins to discretise each action dimension into

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the bin it should be in and compute independent cross ent losses per action dim
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size, y_dim x n_bins
        loss = 0.0
        y_batch = torch.clip(y_batch, min=-0.99, max=0.99)
        for i in range(self.y_dim):
            idx_start = i * self.n_bins
            idx_end = (i + 1) * self.n_bins
            y_pred_dim = y_pred[:, idx_start:idx_end]
            y_true_dim_continuous = y_batch[:, i]
            # now find which bin y_true_dim_continuous is in
            # 1) convert from [-1,1] to [0, n_bins]
            y_true_dim_continuous += 1
            y_true_dim_continuous = y_true_dim_continuous / 2 * self.n_bins
            # 2) round _down_ to nearest integer
            y_true_dim_label = torch.floor(y_true_dim_continuous).long()

            # note that torch's crossent expects logits
            loss += self.loss_crossent(y_pred_dim, y_true_dim_label)
        return loss

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_output = torch.zeros((x_batch.shape[0], self.y_dim))  # set this up
        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)  # these are logits
        for i in range(self.y_dim):
            idx_start = i * self.n_bins
            idx_end = (i + 1) * self.n_bins
            y_pred_dim = y_pred[:, idx_start:idx_end]

            # 1) get class
            if sample_type == "argmax":
                class_idx = torch.argmax(y_pred_dim, dim=-1)
            elif sample_type == "probabilistic":
                # pass through softmax and sample
                y_pred_dim_probs = nn.functional.softmax(y_pred_dim, dim=-1)
                class_idx = torch.squeeze(torch.multinomial(y_pred_dim_probs, num_samples=1))
            # 2) do reverse of scaling, so now [0, n_bins] -> [-1, 1]
            y_output[:, i] = (((class_idx + 0.5) / self.n_bins) * 2) - 1

        return y_output


class Model_Cond_Kmeans(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, kmeans_model):
        super(Model_Cond_Kmeans, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.kmeans_model = kmeans_model

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the kmeans bin it should be in and compute cross ent losses
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size, n_clusters

        # figure out which kmeans bin it should be in
        y_true_label = torch.Tensor(self.kmeans_model.predict(y_batch.cpu())).to(self.device).long()
        return self.loss_crossent(y_pred, y_true_label)

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)  # these are logits
        # 1) get class
        if sample_type == "argmax":
            class_idx = torch.argmax(y_pred, dim=-1)
        elif sample_type == "probabilistic":
            # pass through softmax and sample
            y_pred_probs = nn.functional.softmax(y_pred, dim=-1)
            class_idx = torch.squeeze(torch.multinomial(y_pred_probs, num_samples=1))

        # 2) convert from class to kmeans centroid
        y_output = torch.index_select(
            torch.tensor(self.kmeans_model.cluster_centers_).to(self.device),
            dim=0,
            index=class_idx,
        )

        return y_output


class Model_Cond_BeT(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, kmeans_model):
        super(Model_Cond_BeT, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.loss_crossent = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()
        self.kmeans_model = kmeans_model
        self.n_k = self.kmeans_model.n_clusters

    def loss_on_batch(self, x_batch, y_batch):
        # y_batch comes in continuous
        # we work out the kmeans bin it should be in and compute cross ent losses
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
        y_t = y_batch * 0.0
        y_pred = self.nn_model(y_t, x_batch, _ts, context_mask)
        # y_pred is shape: batch_size,
        y_pred_label = y_pred[:, :self.n_k]

        # figure out which kmeans bin it should be in
        y_true_label = torch.Tensor(self.kmeans_model.predict(y_batch.cpu())).to(self.device).long()

        # now add in the residual
        # y_batch is shape [n_batch, y_dim]
        # y_pred is shape [n_batch, n_k + (y_dim*n_k)]
        # first find chunk of y_pred corresponding to y_true_label
        y_pred_residual_all = y_pred[:, self.n_k:].view(y_pred.shape[0], self.n_k, self.y_dim)  # batch_size, n_k, y_dim
        y_pred_residual = y_pred_residual_all.index_select(1, y_true_label)  # [batch_size,batch_size,y_dim]
        y_pred_residual = torch.diagonal(y_pred_residual, offset=0, dim1=0, dim2=1).T  # I think this is right, may need to check again

        # compute true residual
        K_centers = torch.Tensor(self.kmeans_model.cluster_centers_).to(self.device)  # n_k, y_dim
        y_true_label_center = K_centers.index_select(0, y_true_label)
        y_true_residual = y_batch - y_true_label_center
        return self.loss_crossent(y_pred_label, y_true_label) + 100 * self.loss_mse(y_true_residual, y_pred_residual)

    def sample(self, x_batch, sample_type="probabilistic"):
        # sample_type can be 'argmax' or 'probabilistic'
        # argmax selects most probable class,
        # probabilistic samples via softmax probs
        # "we first sample an action center according to the predicted bin centerprobabilities on thetthindex.
        # Once we have chosen an action centerAt,j, we add the correspondingresidual action〈ˆa(j)t〉to it
        # to recover a predicted continuous actionˆat=At,j+〈ˆa(j)t〉"
        n_sample = x_batch.shape[0]
        _ts = torch.zeros((n_sample, 1)).to(self.device)
        y_shape = (n_sample, self.y_dim)
        y_i = torch.zeros(y_shape).to(self.device)
        context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        y_pred = self.nn_model(y_i, x_batch, _ts, context_mask)
        y_pred_label = y_pred[:, :self.n_k]  # these are logits
        # 1) get class
        if sample_type == "argmax":
            class_idx = torch.argmax(y_pred_label, dim=-1)
        elif sample_type == "probabilistic":
            # pass through softmax and sample
            y_pred_probs = nn.functional.softmax(y_pred_label, dim=-1)
            class_idx = torch.squeeze(torch.multinomial(y_pred_probs, num_samples=1))

        # 2) convert from class to kmeans centroid
        K_centers = torch.Tensor(self.kmeans_model.cluster_centers_).to(self.device)  # n_k, y_dim
        y_pred_label_center = K_centers.index_select(dim=0, index=class_idx)

        # 3) add on residual
        y_pred_residual_all = y_pred[:, self.n_k:].view(y_pred.shape[0], self.n_k, self.y_dim)  # batch_size, n_k, y_dim
        y_pred_residual = y_pred_residual_all.index_select(1, class_idx)  # [batch_size,batch_size,y_dim]
        y_pred_residual = torch.diagonal(y_pred_residual, offset=0, dim1=0, dim2=1).T  # I think this is right, may need to check again

        # 4) add center bin and residual
        y_output = y_pred_label_center + y_pred_residual
        # y_output = y_pred_residual
        return y_output


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                x = x + x2
            else:
                x = x1 + x2
            return x / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x


class Model_cnn_mlp(nn.Module):
    def __init__(self, x_shape, n_hidden, y_dim, embed_dim, net_type, output_dim=None):
        super(Model_cnn_mlp, self).__init__()

        self.x_shape = x_shape
        self.n_hidden = n_hidden
        self.y_dim = y_dim
        self.embed_dim = embed_dim
        self.n_feat = 64
        self.net_type = net_type

        if output_dim is None:
            self.output_dim = y_dim  # by default, just output size of action space
        else:
            self.output_dim = output_dim  # sometimes overwrite, eg for discretised, mean/variance, mixture density models

        # set up CNN for image
        self.conv_down1 = nn.Sequential(
            ResidualConvBlock(self.x_shape[-1], self.n_feat, is_res=True),
            nn.MaxPool2d(2),
        )
        self.conv_down3 = nn.Sequential(
            ResidualConvBlock(self.n_feat, self.n_feat * 2, is_res=True),
            nn.MaxPool2d(2),
        )
        self.imageembed = nn.Sequential(nn.AvgPool2d(8))

        cnn_out_dim = self.n_feat * 2  # how many features after flattening -- WARNING, will have to adjust this for diff size input resolution
        # it is the flattened size after CNN layers, and average pooling

        # then once have flattened vector out of CNN, just feed into previous Model_mlp_diff_embed
        self.nn_downstream = Model_mlp_diff_embed(
            cnn_out_dim,
            self.n_hidden,
            self.y_dim,
            self.embed_dim,
            self.output_dim,
            is_dropout=False,
            is_batch=False,
            activation="relu",
            net_type=self.net_type,
            use_prev=False,
        )

    def forward(self, y, x, t, context_mask, x_embed=None):
        # torch expects batch_size, channels, height, width
        # but we feed in batch_size, height, width, channels

        if x_embed is None:
            x_embed = self.embed_context(x)
        else:
            # otherwise, we already extracted x_embed
            # e.g. outside of sampling loop
            pass

        return self.nn_downstream(y, x_embed, t, context_mask)

    def embed_context(self, x):
        x = x.permute(0, 3, 2, 1)
        x1 = self.conv_down1(x)
        x3 = self.conv_down3(x1)  # [batch_size, 128, 35, 18]
        # c3 is [batch size, 128, 4, 4]
        x_embed = self.imageembed(x3)
        # c_embed is [batch size, 128, 1, 1]
        x_embed = x_embed.view(x.shape[0], -1)
        # c_embed is now [batch size, 128]
        return x_embed


class Model_Cond_EBM(nn.Module):
    def __init__(self, nn_model, device, x_dim, y_dim, n_counter_egs=256, ymin=-1, ymax=1, n_samples=4096, n_iters=3, stddev=0.33, K=0.5, sample_mode="derivative_free"):
        super(Model_Cond_EBM, self).__init__()
        self.nn_model = nn_model
        self.device = device
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_counter_egs = n_counter_egs  # how many counter examples to draw
        self.ymin = ymin  # upper and lower bounds of y
        self.ymax = ymax

        # params used for sampling
        self.n_samples = n_samples
        self.n_iters = n_iters
        self.stddev = stddev
        self.K = K
        self.sample_mode = sample_mode

        # langevin sampling
        self.l_noise_scale = 0.5
        self.l_coeff_start = 0.5
        self.l_coeff_end = 0.005
        self.l_delta_clip = 0.5  # not used currently
        self.l_M = 1
        self.n_mcmc_iters = 40
        self.l_n_sampleloops = 2  # 1 or 2 (only inference time)
        self.l_overwrite = False

    def loss_on_batch(self, x_batch, y_batch, extract_embedding=True):
        # following appendix B, Algorithm 1 of Implicit Behavioral Cloning
        batch_size = x_batch.shape[0]
        loss = 0

        # first need to make batchsize much larger
        y_batch = y_batch.repeat(self.n_counter_egs + 1, 1)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)
            x_embed = x_embed.repeat(self.n_counter_egs + 1, 1)
        else:
            if len(x_batch.shape) == 2:
                x_batch = x_batch.repeat(self.n_counter_egs + 1, 1)
            else:
                x_batch = x_batch.repeat(self.n_counter_egs + 1, 1, 1, 1)

        # unused inputs
        _ts = torch.zeros((y_batch.shape[0], 1)).to(self.device)
        context_mask = torch.zeros(y_batch.shape[0]).to(self.device)

        if self.sample_mode == "langevin":
            l_noise_scale = self.l_noise_scale
            l_coeff_start = self.l_coeff_start
            n_mcmc_iters = self.n_mcmc_iters

            y_samples = y_batch[self.n_counter_egs:, :]

            # random init
            y_samples = torch.rand(y_samples.size()).to(self.device) * (self.ymax - self.ymin) + self.ymin

            # run mcmc chain
            for i in range(n_mcmc_iters):

                l_coeff = self.l_coeff_end + l_coeff_start * (1 - (i / n_mcmc_iters)) ** 2
                y_samples.requires_grad = True

                if extract_embedding:
                    y_pred = self.nn_model(
                        y_samples, x_batch[self.n_counter_egs:], _ts[self.n_counter_egs:], context_mask[self.n_counter_egs:], x_embed[self.n_counter_egs:]
                    )  # forward pass
                else:
                    y_pred = self.nn_model(y_samples, x_batch[self.n_counter_egs:], _ts[self.n_counter_egs:], context_mask[self.n_counter_egs:])  # forward pass

                y_pred_grad = torch.autograd.grad(y_pred, y_samples, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]  # compute gradients
                delta_action = 0.5 * y_pred_grad + torch.randn(size=y_samples.size()).to(self.device) * l_noise_scale
                y_samples = y_samples - l_coeff * (delta_action)
                y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                y_samples = y_samples.detach()

            # these form negative samples
            y_batch[self.n_counter_egs:, :] = y_samples

            # B.3.1 gradient penalty
            loss_grad = torch.maximum(torch.zeros_like(y_pred_grad[:, 0]).to(self.device), (torch.linalg.norm(y_pred_grad, dim=1, ord=np.inf) - self.l_M)) ** 2
            loss += torch.mean(loss_grad)
        else:
            # draw counter-examples from U(y_min, y_max)
            y_batch[self.n_counter_egs:, :] = torch.rand(y_batch[self.n_counter_egs:, :].size()) * (self.ymax - self.ymin) + self.ymin
            # (note we now use the y input from the diffusion model again)

        # forward pass
        if extract_embedding:
            y_pred = self.nn_model(y_batch, x_batch, _ts, context_mask, x_embed)
        else:
            y_pred = self.nn_model(y_batch, x_batch, _ts, context_mask)  # (n_batch x (n_counter_egs+1), 1)

        # y_pred comes out in a vector of size (n_batch x n_counter_egs, 1), we need to reshape this
        y_pred_reshape = y_pred.view(self.n_counter_egs + 1, batch_size).T

        if self.l_overwrite:
            # could overwrite half to be drawn uniformly
            y_pred_reshape[:, 1 + int(self.n_counter_egs / 2)] = (
                torch.rand(y_pred_reshape[:, 1 + int(self.n_counter_egs / 2)].size()) * (self.ymax - self.ymin) + self.ymin
            )

        loss_NCE = -(-y_pred_reshape[:, 0] - torch.logsumexp(-y_pred_reshape, dim=1))
        loss += torch.mean(loss_NCE)

        return loss

    def sample(self, x_batch, extract_embedding=True):
        batch_size = x_batch.shape[0]
        # note that n_samples is only used derivative_free, it refers to the population pool
        # NOT the number of samples to return
        n_samples = self.n_samples
        n_iters = self.n_iters
        stddev = self.stddev
        K = self.K

        if self.sample_mode == "derivative_free":
            # this follows algorithm 1 in appendix B1 of Implicit BC
            y_samples = torch.rand((n_samples * batch_size, self.y_dim)).to(self.device) * (self.ymax - self.ymin) + self.ymin
            if extract_embedding:
                x_embed = self.nn_model.embed_context(x_batch)
                x_embed = x_embed.repeat(n_samples, 1)
                x_size_0 = x_embed.shape[0]
            else:
                if len(x_batch.shape) == 2:
                    x_batch = x_batch.repeat(n_samples, 1)
                else:
                    x_batch = x_batch.repeat(n_samples, 1, 1, 1)
                x_size_0 = x_batch.shape[0]
            for i in range(n_iters):
                # compute energies
                _ts = torch.zeros((x_size_0, 1)).to(self.device)
                context_mask = torch.zeros(x_size_0).to(self.device)
                if extract_embedding:
                    y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask, x_embed)
                else:
                    y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask)
                y_pred_reshape = y_pred.view(n_samples, batch_size).T

                # softmax
                y_probs_reshape = torch.nn.functional.softmax(-y_pred_reshape, dim=1)

                y_samples_reshape = torch.permute(y_samples.view(n_samples, batch_size, self.y_dim), (1, 0, 2))
                if i < n_iters - 1:  # don't want to do this for last iteration
                    # loop over individual samples here
                    for j in range(batch_size):
                        idx_sample = torch.multinomial(y_probs_reshape[j, :], n_samples, replacement=True)
                        y_samples[j * n_samples:(j + 1) * n_samples] = y_samples_reshape[j, idx_sample]
                    y_samples = y_samples + torch.randn(y_samples.size()).to(self.device) * stddev
                    y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                    stddev = stddev * K  # shrink sampling scale

            y_idx = torch.argmin(y_pred_reshape, dim=-1)  # same as doing argmax over probs
            y_output = torch.diagonal(torch.index_select(y_samples_reshape, dim=1, index=y_idx), dim1=0, dim2=1).T
        elif self.sample_mode == "langevin":
            l_noise_scale = self.l_noise_scale
            l_coeff_start = self.l_coeff_start
            l_coeff_end = self.l_coeff_end
            n_mcmc_iters = self.n_mcmc_iters

            y_samples = torch.rand((batch_size, self.y_dim)).to(self.device) * (self.ymax - self.ymin) + self.ymin

            _ts = torch.zeros((x_batch.shape[0], 1)).to(self.device)
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

            if extract_embedding:
                x_embed = self.nn_model.embed_context(x_batch)

            for j in range(self.l_n_sampleloops):
                # run mcmc chain
                for i in range(n_mcmc_iters):
                    if j == 0:
                        l_coeff = self.l_coeff_end + l_coeff_start * (1 - (i / n_mcmc_iters)) ** 2
                    else:
                        l_coeff = l_coeff_end
                    y_samples.requires_grad = True

                    if extract_embedding:
                        y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask, x_embed)
                    else:
                        y_pred = self.nn_model(y_samples, x_batch, _ts, context_mask)  # forward pass
                    y_pred_grad = torch.autograd.grad(y_pred, y_samples, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]  # compute gradients
                    delta_action = 0.5 * y_pred_grad + torch.randn(size=y_samples.size()).to(self.device) * l_noise_scale
                    y_samples = y_samples - l_coeff * (delta_action)
                    y_samples = torch.clip(y_samples, min=self.ymin, max=self.ymax)
                    y_samples = y_samples.detach()
                y_output = y_samples

        return y_output
