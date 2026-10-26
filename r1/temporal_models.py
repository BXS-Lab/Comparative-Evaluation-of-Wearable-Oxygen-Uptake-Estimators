"""Temporal architectures (ported unchanged in structure from experiments/tuning/optuna_temporal_tuning.py).

TFT is a simplified variant (softmax variable selection, one LSTM layer, one multi-head attention layer, GRN);
PatchTST is a PatchTST-style patch Transformer without RevIN, with channel-independent (CI) and
channel-dependent (CD) variants. All output a single value (V̇O₂ at the last window step)."""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUModel(LSTMModel):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__(input_size, hidden_size, num_layers, dropout)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)


class _Chomp(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, :-self.s].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_in, n_out, k, padding=pad, dilation=dilation)), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_out, n_out, k, padding=pad, dilation=dilation)), _Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.relu(self.net(x) + res)


def tcn_receptive_field(num_levels, kernel_size):
    return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_levels, base_channels, kernel_size, dropout):
        super().__init__()
        chans = [min(128, base_channels * (2 ** min(i, 2))) for i in range(num_levels)]
        layers = [_TemporalBlock(input_size if i == 0 else chans[i - 1], c, kernel_size, 2 ** i, dropout)
                  for i, c in enumerate(chans)]
        self.net = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(chans[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.net(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


class TFTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super().__init__()
        self.var_sel = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_size, input_size), nn.Softmax(dim=-1))
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.grn = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        h, _ = self.lstm(self.embed(x * self.var_sel(x)))
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.fc(self.ln(a + self.grn(a))[:, -1, :]).squeeze(-1)


def patch_count(seq_len, patch_len):
    stride = max(4, patch_len // 2)
    return (seq_len - patch_len) // stride + 1, stride


class PatchTSTModel(nn.Module):
    def __init__(self, input_size, seq_len, patch_len, d_model, nhead, num_layers, dropout, channel_mode):
        super().__init__()
        self.ci = channel_mode == "CI"
        self.patch_len = patch_len
        n_patches, self.stride = patch_count(seq_len, patch_len)
        in_dim = patch_len if self.ci else patch_len * input_size
        self.embed = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers, enable_nested_tensor=False)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        B, L, C = x.shape
        p = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, C, patch_len)
        if self.ci:
            p = p.permute(0, 2, 1, 3).reshape(B * C, p.shape[1], self.patch_len)
            z = self.tr(self.embed(p) + self.pos).mean(1).reshape(B, C, -1).mean(1)
        else:
            p = p.permute(0, 1, 3, 2).reshape(B, p.shape[1], self.patch_len * C)
            z = self.tr(self.embed(p) + self.pos).mean(1)
        return self.fc(z).squeeze(-1)
