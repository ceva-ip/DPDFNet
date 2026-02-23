import torch
from torch import nn, Tensor
from torch.nn import Module
from functools import partial
from typing import List, Optional, Tuple

from streaming.layers import Stft, Istft, MagNorm48, \
    GroupedLinearEinsum, GroupedLinear, SqueezedGRU_S, Conv2dNormAct, \
    ConvTranspose2dNormAct, SpecNorm48, SubPixelConv2dNormAct, DPRNN, CyclicBuffer
from model.utils import as_real, to_db, get_wnorm, vorbis_window, erb_filter_banks
from streaming.utils import get_mag
from streaming import multiframe as MF

PI = 3.1415926535897932384626433


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class MagnitudeMask(nn.Module):
    """Streaming mask for 48 kHz HR path where mask is already per-frequency-bin."""
    def __init__(self, freq_bins: int):
        super().__init__()
        self.spec_buffer = CyclicBuffer(
            # [B, 1, T, F, 2]
            shape=[1, 1, 1, freq_bins, 2],
            time_steps=1,
            delay_frames=2,
            time_dim=2,
        )

    def state_size(self) -> int:
        return self.spec_buffer.state_size()

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in MagnitudeMask: expected {self.state_size()}, got {state.numel()}")
            return state
        return self.spec_buffer.initial_state(device=device, dtype=dtype)

    def forward(
            self,
            spec: Tensor,
            mask: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ):
        # spec: [B, 1, T, F, 2], mask: [B, 1, T, F]
        if mask.shape[-1] != spec.shape[-2]:
            raise ValueError(f"Mask/spec frequency mismatch: mask F={mask.shape[-1]}, spec F={spec.shape[-2]}")
        mask = mask.unsqueeze(-1)
        if state is None:
            return self.spec_buffer(spec) * mask
        buffered_spec, state_out, offset = self.spec_buffer(spec, state=state, offset=offset)
        return buffered_spec * mask, state_out, offset


class Encoder(Module):
    def __init__(
            self,
            n_fft: int,
            emb_dim: int,
            nb_erb: int,
            nb_df: int,
            conv_ch: int,
            conv_kernel_inp: Tuple[int, int],
            conv_kernel: Tuple[int, int],
            enc_concat: bool,
            emb_hidden_dim: int,
            enc_lin_groups: int,
            emb_num_layers: int,
            lin_groups: int,
            emb_gru_skip_enc: str = 'none',
            stateful: bool = False,
            group_linear_type: str = 'einsum',
            point_wise_type: str = 'cnn',
            group_gru: int = 1,
            separable_first_conv: bool = True,
            lsnr_min: float = -15.,
            lsnr_max: float = 35.,
            dprnn_num_blocks: int = 0,
    ):
        super().__init__()
        assert nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        # --- Conv0 Buffers --- #
        if conv_kernel_inp[0] > 1:
            self.erb_conv0_buffer = CyclicBuffer(
                # [B, C, T, F]
                shape=[1, 1, 1, n_fft // 2 + 1],
                time_steps=conv_kernel_inp[0],
                delay_frames=0,
                time_dim=2
            )
            self.df_conv0_buffer = CyclicBuffer(
                # [B, C, T, F_df]
                shape=[1, 2, 1, nb_df],
                time_steps=conv_kernel_inp[0],
                delay_frames=0,
                time_dim=2
            )
        else:
            self.erb_conv0_buffer = nn.Identity()
            self.df_conv0_buffer = nn.Identity()

        self.erb_conv0 = Conv2dNormAct(
            in_ch=1,
            out_ch=conv_ch,
            kernel_size=conv_kernel_inp,
            bias=False,
            separable=separable_first_conv,
            point_wise_type=point_wise_type,
        )
        conv_layer = partial(
            Conv2dNormAct,
            in_ch=conv_ch,
            out_ch=conv_ch,
            kernel_size=conv_kernel,
            bias=False,
            separable=True,
            point_wise_type=point_wise_type,
        )
        self.erb_conv1 = conv_layer(fstride=3)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=2)
        self.df_conv0 = Conv2dNormAct(
            in_ch=2,
            out_ch=conv_ch,
            kernel_size=conv_kernel_inp,
            bias=False,
            separable=separable_first_conv,
            point_wise_type=point_wise_type,
        )
        self.df_conv1 = conv_layer(fstride=2)

        self.dprnn_erb = DPRNN(
            num_feat=n_fft // 2 // 3 // 2 // 2,
            ch_in=conv_ch,
            hidden_dim=conv_ch,
            ch_out=conv_ch,
            num_blocks=dprnn_num_blocks,
            stateful=stateful
        ) if dprnn_num_blocks > 0 else nn.Identity()

        self.dprnn_df = DPRNN(
            num_feat=nb_df // 2,
            ch_in=conv_ch,
            hidden_dim=conv_ch,
            ch_out=conv_ch,
            num_blocks=dprnn_num_blocks,
            stateful=stateful
        ) if dprnn_num_blocks > 0 else nn.Identity()
        self.erb_bins = nb_erb
        self.emb_in_dim = emb_dim
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = conv_ch * nb_erb // 4
        group_linear_layer = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        erb_fc_emb = group_linear_layer(
            input_size=conv_ch * (n_fft // 2 // 3 // 2 // 2),
            hidden_size=self.emb_in_dim,
            groups=enc_lin_groups
        )
        df_fc_emb = group_linear_layer(
            input_size=conv_ch * nb_df // 2,
            hidden_size=self.emb_in_dim,
            groups=enc_lin_groups
        )
        self.erb_fc_emb = nn.Sequential(erb_fc_emb, nn.ReLU(inplace=True))
        self.df_fc_emb = nn.Sequential(df_fc_emb, nn.ReLU(inplace=True))
        if enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = emb_num_layers
        if emb_gru_skip_enc == "none":
            skip_op = None
        elif emb_gru_skip_enc == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(
                group_linear_layer,
                input_size=self.emb_out_dim,
                hidden_size=self.emb_out_dim,
                groups=lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            gru_skip_op=skip_op,
            linear_groups=lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=group_linear_layer,
            stateful=stateful,
        )

        # mic/accel signal quality estimation:
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def state_size(self) -> int:
        size = 0
        for module in (self.erb_conv0_buffer, self.dprnn_erb, self.df_conv0_buffer, self.dprnn_df, self.emb_gru):
            if hasattr(module, "state_size"):
                size += module.state_size()
        return size

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in Encoder: expected {self.state_size()}, got {state.numel()}")
            return state
        chunks: List[Tensor] = []
        for module in (self.erb_conv0_buffer, self.dprnn_erb, self.df_conv0_buffer, self.dprnn_df, self.emb_gru):
            if hasattr(module, "initial_state"):
                chunks.append(module.initial_state(device=device, dtype=dtype))
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=dtype, device=device)

    def forward(
            self,
            feat_erb: Tensor,
            feat_spec: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ):
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        # b, _, t, _ = feat_erb.shape
        state_out_chunks: List[Tensor] = []

        if state is None or not hasattr(self.erb_conv0_buffer, "state_size"):
            feat_erb_buffered = self.erb_conv0_buffer(feat_erb)
        else:
            feat_erb_buffered, erb_conv0_state_out, offset = self.erb_conv0_buffer(feat_erb, state=state, offset=offset)
            state_out_chunks.append(erb_conv0_state_out)
        e0 = self.erb_conv0(feat_erb_buffered[..., :-1])  # [B, C, T, F[:-1]]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/3]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/6]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/12]
        if state is None or not hasattr(self.dprnn_erb, "state_size"):
            e3_dprnn = self.dprnn_erb(e3)
        else:
            e3_dprnn, dprnn_erb_state_out, offset = self.dprnn_erb(e3, state=state, offset=offset)
            state_out_chunks.append(dprnn_erb_state_out)

        if state is None or not hasattr(self.df_conv0_buffer, "state_size"):
            feat_spec_buffered = self.df_conv0_buffer(feat_spec)
        else:
            feat_spec_buffered, df_conv0_state_out, offset = self.df_conv0_buffer(feat_spec, state=state, offset=offset)
            state_out_chunks.append(df_conv0_state_out)
        c0 = self.df_conv0(feat_spec_buffered)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc/2]
        if state is None or not hasattr(self.dprnn_df, "state_size"):
            c1_dprnn = self.dprnn_df(c1)
        else:
            c1_dprnn, dprnn_df_state_out, offset = self.dprnn_df(c1, state=state, offset=offset)
            state_out_chunks.append(dprnn_df_state_out)
        cemb = c1_dprnn.permute(0, 2, 3, 1).flatten(1)  # [B, -1]
        cemb = self.df_fc_emb(cemb)
        emb = e3_dprnn.permute(0, 2, 3, 1).flatten(1)  # [B, C * F]
        emb = self.erb_fc_emb(emb)
        emb = self.combine(emb, cemb)
        if state is None:
            emb = self.emb_gru(emb)
        else:
            emb, emb_gru_state_out, offset = self.emb_gru(emb, state=state, offset=offset)
            state_out_chunks.append(emb_gru_state_out)
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset
        if state is None:
            return e0, e1, e2, e3, emb, c0, lsnr
        state_out = torch.cat(state_out_chunks, dim=0) if state_out_chunks else state.new_zeros(0)
        return e0, e1, e2, e3, emb, c0, lsnr, state_out, offset


class ErbDecoder(Module):
    def __init__(
            self,
            n_fft: int,
            emb_dim: int,
            nb_erb: int,
            conv_ch: int,
            conv_kernel: Tuple[int, int],
            convt_kernel: Tuple[int, int],
            emb_num_layers: int,
            emb_hidden_dim: int,
            lin_groups: int,
            enc_lin_groups: int,
            emb_gru_skip: str = 'none',
            stateful: bool = False,
            group_linear_type: str = 'einsum',
            upsample_conv_type: str = 'transpose',
            point_wise_type: str = 'cnn',
            group_gru: int = 1,
    ):
        super().__init__()
        assert nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_in_dim = emb_dim
        self.emb_dim = emb_hidden_dim
        self.emb_out_dim = emb_dim

        group_linear_layer = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        if emb_gru_skip == "none":
            skip_op = None
        elif emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif emb_gru_skip == "groupedlinear":
            skip_op = partial(
                group_linear_layer,
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                groups=lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=emb_num_layers,
            gru_skip_op=skip_op,
            linear_groups=lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=group_linear_layer,
            stateful=stateful,
        )
        upsample_conv_layer = ConvTranspose2dNormAct if upsample_conv_type == 'transpose' else SubPixelConv2dNormAct
        tconv_layer = partial(
            upsample_conv_layer,
            kernel_size=convt_kernel,
            bias=False,
            separable=True,
            point_wise_type=point_wise_type,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
            point_wise_type=point_wise_type,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt3 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv2p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(conv_ch, conv_ch, fstride=2)
        self.conv1p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(conv_ch, conv_ch, fstride=3)
        self.conv0p = conv_layer(conv_ch, conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            conv_ch, 1, kernel_size=conv_kernel, activation_layer=nn.Sigmoid
        )
        erb_fc_emb = group_linear_layer(
            input_size=emb_dim,
            hidden_size=conv_ch * (n_fft // 2 // 3 // 2 // 2),
            groups=enc_lin_groups
        )
        self.erb_fc_emb = nn.Sequential(erb_fc_emb, nn.ReLU(inplace=True))

    def state_size(self) -> int:
        return self.emb_gru.state_size() if hasattr(self.emb_gru, "state_size") else 0

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in ErbDecoder: expected {self.state_size()}, got {state.numel()}")
            return state
        if hasattr(self.emb_gru, "initial_state"):
            return self.emb_gru.initial_state(device=device, dtype=dtype)
        return torch.zeros(0, dtype=dtype, device=device)

    def forward(
            self,
            emb: Tensor,
            e3: Tensor,
            e2: Tensor,
            e1: Tensor,
            e0: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ):
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        if state is None:
            emb = self.emb_gru(emb)
            emb_state_out = None
        else:
            emb, emb_state_out, offset = self.emb_gru(emb, state=state, offset=offset)
        emb = self.erb_fc_emb(emb)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        m = torch.nn.functional.pad(m, pad=(0, 1, 0, 0), mode='reflect')
        if state is None:
            return m
        state_out = emb_state_out if emb_state_out is not None else state.new_zeros(0)
        return m, state_out, offset


class DfOutputReshapeMF(nn.Module):
    """Coefficients output reshape for multiframe/MultiFrameModule

    Requires input of shape B, C, T, F, 2.
    """

    def __init__(self, df_order: int, df_bins: int):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins

    def forward(self, coefs: Tensor) -> Tensor:
        # [B, T, F, O*2] -> [B, O, T, F, 2]
        new_shape = list(coefs.shape)
        new_shape[-1] = -1
        new_shape.append(2)
        coefs = coefs.view(new_shape)
        coefs = coefs.permute(0, 3, 1, 2, 4)
        return coefs


class DfDecoder(Module):
    def __init__(
            self,
            nb_erb: int,
            nb_df: int,
            conv_ch: int,
            df_hidden_dim: int,
            emb_hidden_dim: int,
            df_order: int,
            df_num_layers: int,
            df_pathway_kernel_size_t: int,
            lin_groups: int,
            df_gru_skip: str = 'groupedlinear',
            stateful: bool = False,
            group_linear_type: str = 'einsum',
            point_wise_type: str = 'cnn',
            group_gru: int = 1,
    ):
        super().__init__()
        layer_width = conv_ch

        self.emb_in_dim = conv_ch * nb_erb // 4
        self.emb_dim = df_hidden_dim

        self.df_n_hidden = df_hidden_dim
        self.df_n_layers = df_num_layers
        self.df_order = df_order
        self.df_bins = nb_df
        self.df_out_ch = df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False, point_wise_type=point_wise_type,)
        kt = df_pathway_kernel_size_t
        self.df_convp_buffer = CyclicBuffer(
            # [B, C, T, F_df]
            shape=[1, conv_ch, 1, nb_df],
            time_steps=kt,
            delay_frames=0,
            time_dim=2
            )
        self.df_convp = conv_layer(layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1))

        group_linear_layer = GroupedLinearEinsum if group_linear_type == 'einsum' else GroupedLinear
        self.df_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
            group_linear_layer=group_linear_layer,
            stateful=stateful,
        )
        df_gru_skip = df_gru_skip.lower()
        assert df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: Optional[nn.Module]
        if df_gru_skip == "none":
            self.df_skip = None
        elif df_gru_skip == "identity":
            assert emb_hidden_dim == df_hidden_dim, "Dimensions do not match"
            self.df_skip = nn.Identity()
        elif df_gru_skip == "groupedlinear":
            self.df_skip = group_linear_layer(self.emb_in_dim, self.emb_dim, groups=lin_groups)
        else:
            raise NotImplementedError()
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        df_out = group_linear_layer(self.df_n_hidden, out_dim, groups=lin_groups)
        self.df_out = nn.Sequential(df_out, nn.Tanh())

    def state_size(self) -> int:
        size = 0
        if hasattr(self.df_gru, "state_size"):
            size += self.df_gru.state_size()
        if hasattr(self.df_convp_buffer, "state_size"):
            size += self.df_convp_buffer.state_size()
        return size

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in DfDecoder: expected {self.state_size()}, got {state.numel()}")
            return state
        chunks: List[Tensor] = []
        if hasattr(self.df_gru, "initial_state"):
            chunks.append(self.df_gru.initial_state(device=device, dtype=dtype))
        if hasattr(self.df_convp_buffer, "initial_state"):
            chunks.append(self.df_convp_buffer.initial_state(device=device, dtype=dtype))
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=dtype, device=device)

    def forward(
            self,
            emb: Tensor,
            c0: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ):
        b = emb.shape[0]
        state_out_chunks: List[Tensor] = []

        if state is None:
            c = self.df_gru(emb)  # [B, T, H], H: df_n_hidden
        else:
            c, df_gru_state_out, offset = self.df_gru(emb, state=state, offset=offset)  # [B, T, H], H: df_n_hidden
            state_out_chunks.append(df_gru_state_out)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        if state is None:
            c0_buffered = self.df_convp_buffer(c0)
        else:
            c0_buffered, df_convp_state_out, offset = self.df_convp_buffer(c0, state=state, offset=offset)
            state_out_chunks.append(df_convp_state_out)
        c0 = self.df_convp(c0_buffered).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        t = c0.shape[1]
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        if c.dim() == 2:
            if t != 1:
                raise ValueError(f"DfDecoder expected t=1 when df_out is 2D, got t={t}")
            c = c.unsqueeze(1)
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        if state is None:
            return c
        state_out = torch.cat(state_out_chunks, dim=0) if state_out_chunks else state.new_zeros(0)
        return c, state_out, offset


class DPDFNet48HR(Module):
    def __init__(
            self,
            emb_dim: int = 512,
            n_fft: int = 960,
            win_length: float = 0.02,
            hop_length: float = 0.01,
            samplerate: int = 48000,
            freq_df: int = 4800,
            nb_erb: int = 32,
            min_nb_freqs: int = 2,
            erb_to_db: bool = True,
            alpha_norm: float = 0.98,
            conv_ch: int = 64,
            conv_kernel_inp: Tuple[int, int] = (3, 3),
            conv_kernel: Tuple[int, int] = (1, 3),
            convt_kernel: Tuple[int, int] = (1, 3),
            enc_gru_dim: int = 256,
            erb_dec_gru_dim: int = 256,
            df_dec_gru_dim: int = 256,
            enc_lin_groups: int = 32, #32,
            emb_gru_skip_enc: str = 'none',
            lin_groups: int = 16, #16,
            df_order: int = 5,
            df_pathway_kernel_size_t: int = 5,
            df_gru_skip: str = 'groupedlinear',
            df_lookahead: int = 2,
            conv_lookahead: int = 2,
            enc_concat: bool = True,
            emb_num_layers: int = 2,
            df_num_layers: int = 2,
            stateful: bool = False,
            mask_method: str = 'before_df',
            erb_dynamic_var: bool = False,
            norm_stateful: bool = False,
            upsample_conv_type: str = 'subpixel',     # transpose | subpixel
            group_linear_type: str = 'loop',     # einsum | loop
            point_wise_type: str = 'cnn',   # cnn | linear
            group_gru: int = 1,
            separable_first_conv: bool = True,
            lsnr_min: float = -15.,
            lsnr_max: float = 35.,
            dprnn_num_blocks: int = 2,
            **kwargs
    ):

        super().__init__()

        assert upsample_conv_type in ['transpose', 'subpixel']
        assert group_linear_type in ['einsum', 'loop']
        assert point_wise_type in ['cnn', 'linear']

        self.mask_method = mask_method
        self.run_df = True
        self.nb_erb = nb_erb
        self.erb_to_db = erb_to_db
        erb_filters = erb_filter_banks(
            nfft=n_fft,
            low_freq=0,
            fs=samplerate,
            n_filters=nb_erb,
            min_nb_freqs=min_nb_freqs
        )
        erb_filters = torch.tensor(erb_filters, dtype=torch.float32)
        inv_erb_filters = erb_filters.clone().t()
        erb_filters = erb_filters / erb_filters.sum(-1, keepdims=True)

        self.erb_fb: Tensor
        self.erb_inv_fb: Tensor

        self.register_buffer('erb_fb', erb_filters.t())
        self.register_buffer('erb_inv_fb', inv_erb_filters.t())

        window_size = int(win_length * samplerate)
        hop_size = int(hop_length * samplerate)
        self.stft = Stft(
            n_fft=n_fft,
            win_len=window_size,
            hop=hop_size,
            window=vorbis_window(window_size),
            normalized=False,
        )
        self.istft = Istft(
            n_fft_inv=n_fft,
            win_len_inv=window_size,
            hop_inv=hop_size,
            window_inv=vorbis_window(window_size),
            normalized=False,
        )
        self.istft_norm = Istft(
            n_fft_inv=n_fft,
            win_len_inv=window_size,
            hop_inv=hop_size,
            window_inv=vorbis_window(window_size),
            normalized=True,
        )
        self.wnorm = get_wnorm(window_size, hop_size)

        layer_width = conv_ch
        assert nb_erb % 8 == 0, "erb_bins should be divisible by 8"
        self.df_lookahead = df_lookahead
        self.freq_bins: int = n_fft // 2 + 1
        self.nb_df = int((freq_df / (samplerate // 2)) * self.freq_bins)
        self.emb_dim = emb_dim
        self.erb_bins: int = nb_erb
        if conv_lookahead > 0:
            assert conv_lookahead >= df_lookahead
            self.pad_feat = nn.ConstantPad2d((0, 0, -conv_lookahead, conv_lookahead), 0.0)
        else:
            self.pad_feat = nn.Identity()
        if df_lookahead > 0:
            self.pad_spec = nn.ConstantPad3d((0, 0, 0, 0, -df_lookahead, df_lookahead), 0.0)
        else:
            self.pad_spec = nn.Identity()
        self.enc = Encoder(
            n_fft=n_fft,
            emb_dim=emb_dim,
            nb_erb=nb_erb,
            nb_df=self.nb_df,
            conv_ch=conv_ch,
            conv_kernel_inp=conv_kernel_inp,
            conv_kernel=conv_kernel,
            enc_concat=enc_concat,
            emb_hidden_dim=enc_gru_dim,
            enc_lin_groups=enc_lin_groups,
            emb_num_layers=emb_num_layers - 1,
            lin_groups=lin_groups,
            emb_gru_skip_enc=emb_gru_skip_enc,
            stateful=stateful,
            group_linear_type=group_linear_type,
            point_wise_type=point_wise_type,
            group_gru=group_gru,
            separable_first_conv=separable_first_conv,
            lsnr_min=lsnr_min,
            lsnr_max=lsnr_max,
            dprnn_num_blocks=dprnn_num_blocks,
        )
        self.erb_dec = ErbDecoder(
            n_fft=n_fft,
            emb_dim=emb_dim,
            nb_erb=nb_erb,
            conv_ch=conv_ch,
            conv_kernel=conv_kernel,
            convt_kernel=convt_kernel,
            emb_num_layers=emb_num_layers,
            emb_hidden_dim=erb_dec_gru_dim,
            lin_groups=lin_groups,
            enc_lin_groups=enc_lin_groups,
            emb_gru_skip=emb_gru_skip_enc,
            stateful=stateful,
            upsample_conv_type=upsample_conv_type,
            group_linear_type=group_linear_type,
            point_wise_type=point_wise_type,
            group_gru=group_gru,
        )
        self.mask = MagnitudeMask(self.freq_bins)

        self.df_order = df_order
        self.df_op = MF.DF(
            freq_bins=self.freq_bins,
            num_freqs=self.nb_df,
            frame_size=df_order,
            lookahead=self.df_lookahead
        )
        self.df_dec = DfDecoder(
            nb_erb=nb_erb,
            nb_df=self.nb_df,
            conv_ch=conv_ch,
            df_hidden_dim=df_dec_gru_dim,
            emb_hidden_dim=erb_dec_gru_dim,
            df_order=df_order,
            df_num_layers=df_num_layers,
            df_pathway_kernel_size_t=df_pathway_kernel_size_t,
            lin_groups=lin_groups,
            df_gru_skip=df_gru_skip,
            stateful=stateful,
            group_linear_type=group_linear_type,
            point_wise_type='cnn',
            group_gru=group_gru,
        )
        self.df_out_transform = DfOutputReshapeMF(self.df_order, self.nb_df)

        self.run_erb = self.nb_df + 1 < self.freq_bins

        self.erb_norm = MagNorm48(
            num_feat=self.freq_bins,
            alpha=alpha_norm,
            dynamic_var=erb_dynamic_var,
            stateful=norm_stateful
        )
        self.spec_norm = SpecNorm48(
            num_feat=self.nb_df,
            alpha=alpha_norm,
            stateful=norm_stateful
        )

        # print the model's size:
        print('\n' + '-' * 50)
        print(self._count_parameters())
        print('-' * 50 + '\n')

    def state_size(self) -> int:
        return (
            self.erb_norm.state_size()
            + self.spec_norm.state_size()
            + self.enc.state_size()
            + self.erb_dec.state_size()
            + self.df_dec.state_size()
            + self.mask.state_size()
            + self.df_op.state_size()
        )

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in DPDFNet48HR: expected {self.state_size()}, got {state.numel()}")
            return state
        chunks = [
            self.erb_norm.initial_state(device=device, dtype=dtype),
            self.spec_norm.initial_state(device=device, dtype=dtype),
            self.enc.initial_state(device=device, dtype=dtype),
            self.erb_dec.initial_state(device=device, dtype=dtype),
            self.df_dec.initial_state(device=device, dtype=dtype),
            self.mask.initial_state(device=device, dtype=dtype),
            self.df_op.initial_state(device=device, dtype=dtype),
        ]
        return torch.cat(chunks, dim=0)

    def forward(self, spec: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # spec shape: # [B=1, T=1, F, 2]
        expected_state_size = self.state_size()
        if state is None:
            state = self.initial_state(device=spec.device, dtype=torch.float32)
        else:
            if state.ndim != 1:
                raise ValueError(f"state must be 1D [S], got shape {tuple(state.shape)}")
            if state.numel() != expected_state_size:
                raise ValueError(f"state size mismatch: expected {expected_state_size}, got {state.numel()}")
            if state.device != spec.device:
                raise ValueError(f"state device mismatch: expected {spec.device}, got {state.device}")
            if state.dtype != torch.float32:
                raise ValueError(f"state dtype mismatch: expected torch.float32, got {state.dtype}")

        state_out_chunks: List[Tensor] = []
        offset = 0

        spec, feat_erb, feat_spec, feat_state_out, offset = self._feature_extraction(spec, state=state, offset=offset)
        state_out_chunks.append(feat_state_out)

        feat_spec = feat_spec.permute(0, 3, 1, 2)

        e0, e1, e2, e3, emb, c0, lsnr, enc_state_out, offset = self.enc(
            feat_erb, feat_spec, state=state, offset=offset
        )
        state_out_chunks.append(enc_state_out)

        m, erb_dec_state_out, offset = self.erb_dec(emb, e3, e2, e1, e0, state=state, offset=offset)
        state_out_chunks.append(erb_dec_state_out)

        df_coefs, df_dec_state_out, offset = self.df_dec(emb, c0, state=state, offset=offset)
        state_out_chunks.append(df_dec_state_out)
        df_coefs = self.df_out_transform(df_coefs)

        if self.mask_method == 'before_df':
            spec, mask_state_out, offset = self.mask(spec, m, state=state, offset=offset)
            state_out_chunks.append(mask_state_out)
            spec_e, df_state_out, offset = self.df_op(spec.clone(), df_coefs, state=state, offset=offset)
            state_out_chunks.append(df_state_out)
        elif self.mask_method == 'separate':
            spec_m, mask_state_out, offset = self.mask(spec, m, state=state, offset=offset)
            state_out_chunks.append(mask_state_out)
            spec_e, df_state_out, offset = self.df_op(spec.clone(), df_coefs, state=state, offset=offset)
            state_out_chunks.append(df_state_out)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]
        elif self.mask_method == 'after_df':
            spec_e, df_state_out, offset = self.df_op(spec.clone(), df_coefs, state=state, offset=offset)
            state_out_chunks.append(df_state_out)
            spec_e, mask_state_out, offset = self.mask(spec_e, m, state=state, offset=offset)
            state_out_chunks.append(mask_state_out)
        else:
            raise ValueError(f'the mask_method: {self.mask_method} is not exists.')

        spec_e = spec_e.squeeze(1)
        if offset != expected_state_size:
            raise RuntimeError(f"State offset mismatch: consumed {offset}, expected {expected_state_size}")
        state_out = torch.cat(state_out_chunks, dim=0) if state_out_chunks else state.new_zeros(0)
        return spec_e, state_out

    def _count_parameters(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        return f"{self.__class__.__name__}: {num_params / 1e6:.3f}M"

    @torch.no_grad()
    def _feature_extraction(
            self,
            spec: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ):
        """Forward method of DeepFilterNet2.

                Args:
                    spec (Tensor): Spectrum of shape [B=1, T=1, F, 2]

                Returns:
                    spec (Tensor): Spectrum of shape [B, 1, T, F, 2]
                    feat_erb (Tensor): ERB features of shape [B, 1, T, E]
                    feat_spec (Tensor): Complex spectrogram features of shape [B, 1, T, F', 2]
                """
        feat_erb = get_mag(spec)  # (B, T, F)
        if self.erb_to_db:
            feat_erb = to_db(feat_erb)
        feat_spec = spec[..., :self.nb_df, :]     # (B, T, F', 2)

        # normalization
        if state is None:
            feat_erb = self.erb_norm(feat_erb)
            feat_spec = self.spec_norm(feat_spec)
            feat_state_out = None
        else:
            feat_erb, erb_norm_state_out, offset = self.erb_norm(feat_erb, state=state, offset=offset)
            feat_spec, spec_norm_state_out, offset = self.spec_norm(feat_spec, state=state, offset=offset)
            feat_state_out = torch.cat([erb_norm_state_out, spec_norm_state_out], dim=0)

        # reshaping
        spec = as_real(spec.unsqueeze(1))       # [B, 1, T, F, 2]
        feat_erb = feat_erb.unsqueeze(1)        # [B, 1, T, F]

        if state is None:
            return spec, feat_erb, feat_spec
        return spec, feat_erb, feat_spec, feat_state_out, offset

    def apply_stft(self, audio: Tensor) -> Tensor:
        # input shape: float32 (B, #samples)
        # output shape: complex64 (B, T, F)
        audio_pad = nn.functional.pad(audio, (0, self.stft.win_len, 0, 0))
        spec = self.stft(audio_pad).transpose(1, 2)
        spec *= self.wnorm
        return spec

    def apply_istft(self, spec: Tensor) -> Tensor:
        # input shape: complex64 (B, T, F)
        # output shape: float32 (B, #samples)

        if self.training:
            audio = self.istft_norm(spec.transpose(1, 2))
        else:
            audio = self.istft(spec.transpose(1, 2))
            audio /= self.wnorm

        audio = nn.functional.pad(audio[:, self.stft.win_len * 2:], (0, self.stft.win_len * 2, 0, 0))
        return audio


def correct_state_dict(sd: dict) -> dict:
    streaming_sd = dict()
    for k, v in sd.items():
        # 48k HR streaming uses per-bin magnitude mask; checkpoint may still carry ERB inverse FB.
        if k == "mask.erb_inv_fb":
            continue
        if 'dprnn' and 'inter_gru' in k:
            k_tag = k.replace('_l0', '').replace('inter_gru.', 'inter_gru.grucell.')
        elif 'gru.gru' in k:
            l = k[-1] # extract the layer No.
            k_tag = k[:-3] # remoev the suffix "_l0/1"
            k_tag = k_tag.replace('.gru.', f'.gru.{l}.grucell.')
        else:
            k_tag = k
        streaming_sd[k_tag] = v
    return streaming_sd

if __name__ == '__main__':
    import soundfile
    waveform_16k, sr = soundfile.read("./noisy_48khz.wav")
    reference_path = "./enhanced_ref_48khz.wav"
    output_path = "./enhanced_flat_buffer_48khz.wav"
    model = DPDFNet48HR(
        conv_kernel_inp=(3, 3),  # (t, f)
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type='subpixel',  # transpose | subpixel
        group_linear_type='loop',  # einsum | loop
        point_wise_type='cnn',  # cnn | linear
        separable_first_conv=True,
        dprnn_num_blocks=2,
    )
    print(model)

    # load PT weights
    state_dict = torch.load('./model_zoo/checkpoints/dpdfnet2_48khz_hr.pth', weights_only=True, map_location='cpu')
    stream_state_dict = correct_state_dict(state_dict)
    model.load_state_dict(stream_state_dict, strict=True)
    model.eval()

    spec = model.apply_stft(torch.tensor(waveform_48k, dtype=torch.float32)[None, :]) # [B=1, T, F]
    spec = torch.view_as_real(spec) # [B=1, T, F, 2]
    spec_e = []
    state = model.initial_state(dtype=torch.float32)
    with torch.no_grad():
        for t in range(spec.shape[1]):
            out, state = model(spec[:, t:t+1], state)
            spec_e.append(out)
        spec_e = torch.cat(spec_e, dim=1)
    waveform_48k_e = model.apply_istft(spec_e).detach().cpu().numpy().flatten()[:waveform_48k.size]

    soundfile.write(output_path, waveform_48k_e, sr)
    waveform_est, est_sr = soundfile.read(output_path)
    waveform_ref, ref_sr = soundfile.read(reference_path)
    if sr != est_sr:
        raise RuntimeError(f"Sample-rate mismatch: output_file={est_sr}, expected={sr}")
    if sr != ref_sr:
        raise RuntimeError(f"Sample-rate mismatch: output={sr}, reference={ref_sr}")
    waveform_ref = torch.tensor(waveform_ref, dtype=torch.float32).flatten()
    waveform_est = torch.tensor(waveform_est, dtype=torch.float32).flatten()
    if waveform_ref.shape != waveform_est.shape:
        raise RuntimeError(f"Shape mismatch: output={tuple(waveform_est.shape)}, reference={tuple(waveform_ref.shape)}")
    if not torch.allclose(waveform_est, waveform_ref, atol=1e-5, rtol=0.0):
        max_abs = torch.max(torch.abs(waveform_est - waveform_ref)).item()
        raise AssertionError(f"Regression check failed (atol=1e-5). max_abs_err={max_abs:.8f}")
    print("Test passed!")
