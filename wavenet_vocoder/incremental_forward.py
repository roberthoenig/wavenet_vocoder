def incremental_forward(self, initial_input=None, c=None, g=None,
                        T=100, test_inputs=None,
                        tqdm=lambda x: x, softmax=True, quantize=True,
                        log_scale_min=-7.0):
    """Incremental forward step

    Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
    to (B x T x C) internally and fed to the network for each time step.
    Input of each time step will be of shape (B x 1 x C).

    Args:
        initial_input (Tensor): Initial decoder input, (B x C x 1)
        c (Tensor): Local conditioning features, shape (B x C' x T)
        g (Tensor): Global conditioning features, shape (B x C'' or B x C''x 1)
        T (int): Number of time steps to generate.
        test_inputs (Tensor): Teacher forcing inputs (for debugging)
        tqdm (lamda) : tqdm
        softmax (bool) : Whether applies softmax or not
        quantize (bool): Whether quantize softmax output before feeding the
            network output to input for the next time step. TODO: rename
        log_scale_min (float):  Log scale minimum value.

    Returns:
        Tensor: Generated one-hot encoded samples. B x C x T　
            or scaler vector B x 1 x T
    """
    self.clear_buffer()
    B = 1

    # Note: shape should be **(B x T x C)**, not (B x C x T) opposed to
    # batch forward due to linealized convolution
    if test_inputs is not None:
        if self.scalar_input:
            if test_inputs.size(1) == 1:
                test_inputs = test_inputs.transpose(1, 2).contiguous()
        else:
            if test_inputs.size(1) == self.out_channels:
                test_inputs = test_inputs.transpose(1, 2).contiguous()

        B = test_inputs.size(0)
        if T is None:
            T = test_inputs.size(1)
        else:
            T = max(T, test_inputs.size(1))
    # cast to int in case of numpy.int64...
    T = int(T)

    # Global conditioning
    if g is not None:
        if self.embed_speakers is not None:
            g = self.embed_speakers(g.view(B, -1))
            # (B x gin_channels, 1)
            g = g.transpose(1, 2)
            assert g.dim() == 3
    g_btc = _expand_global_features(B, T, g, bct=False)

    # Local conditioning
    if c is not None and self.upsample_conv is not None:
        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)
        assert c.size(-1) == T
    if c is not None and c.size(-1) == T:
        c = c.transpose(1, 2).contiguous()

    outputs = []
    if initial_input is None:
        if self.scalar_input:
            initial_input = torch.zeros(B, 1, 1)
        else:
            initial_input = torch.zeros(B, 1, self.out_channels)
            initial_input[:, :, 127] = 1  # TODO: is this ok?
        # https://github.com/pytorch/pytorch/issues/584#issuecomment-275169567
        if next(self.parameters()).is_cuda:
            initial_input = initial_input.cuda()
    else:
        if initial_input.size(1) == self.out_channels:
            initial_input = initial_input.transpose(1, 2).contiguous()

    current_input = initial_input

    for t in tqdm(range(T)):
        if test_inputs is not None and t < test_inputs.size(1):
            current_input = test_inputs[:, t, :].unsqueeze(1)
        else:
            if t > 0:
                current_input = outputs[-1]

        # Conditioning features for single time step
        ct = None if c is None else c[:, t, :].unsqueeze(1)
        gt = None if g is None else g_btc[:, t, :].unsqueeze(1)

        x = current_input
        x = self.first_conv.incremental_forward(x)
        skips = None
        for f in self.conv_layers:
            x, h = f.incremental_forward(x, ct, gt)
            if self.legacy:
                skips = h if skips is None else (skips + h) * math.sqrt(0.5)
            else:
                skips = h if skips is None else (skips + h)
        x = skips
        for f in self.last_conv_layers:
            try:
                x = f.incremental_forward(x)
            except AttributeError:
                x = f(x)

        # Generate next input by sampling
        if self.scalar_input:
            x = sample_from_discretized_mix_logistic(
                x.view(B, -1, 1), log_scale_min=log_scale_min)
        else:
            x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
            if quantize:
                sample = np.random.choice(
                    np.arange(self.out_channels), p=x.view(-1).data.cpu().numpy())
                x.zero_()
                x[:, sample] = 1.0
        outputs += [x.data]
    # T x B x C
    outputs = torch.stack(outputs)
    # B x C x T
    outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

    self.clear_buffer()
    return outputs