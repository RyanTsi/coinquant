import math
import logging
import numpy as np
import torch
from torch import nn
import copy
from torch import optim
from torch.nn.modules.container import ModuleList
from coinquant.model.basemodel import BaseModel

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    def __init__(
        self,
        d_feat: int = 20,
        d_linear: int = 128,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        early_stop=5,
        optimizer="adam",
        reg=1e-3,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        # set hyper-parameters.
        self.d_linear = d_linear
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(d_feat, d_linear, d_model, nhead, num_layers, dropout, self.device)

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        
        self.criterion = nn.MSELoss()
        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def train_epoch(self, data_loader):
        self.model.train()
        for feature, label in data_loader:
            feature = feature.to(self.device)
            label = label.to(self.device)

            pred = self.model(feature.float())  # .float()
            loss = self.criterion(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():

            for feature, label in data_loader:
                feature = feature.to(self.device)
                label = label.to(self.device)

                pred = self.model(feature.float()) # .float()
                loss = self.criterion(pred, label)

                total_loss += loss.item()

        return total_loss / len(data_loader)

    def fit(
        self,
        train_loader,
        valid_loader,
        evals_result=None,
        save_path=None,
    ):
        if evals_result is None:
            evals_result = {}
        stop_steps = 0
        best_loss = np.inf
        best_epoch = 0
        best_param = copy.deepcopy(self.model.state_dict())
        evals_result['train'] = []
        evals_result['valid']   = []

        # train
        logger.info("training...")

        for step in range(self.n_epochs):
            logger.info("Epoch%d:", step)
            logger.info("training...")
            self.train_epoch(train_loader)
            logger.info("evaluating...")
            train_loss = self.test_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            logger.info("train %.6f, valid %.6f" % (train_loss, val_loss))
            evals_result["train"].append(train_loss)
            evals_result["valid"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    logger.info("early stop")
                    break
        
        logger.info("best loss: %.6lf @ %d" % (best_loss, best_epoch))
        self.model.load_state_dict(best_param)
        self.fitted = True
        
        if save_path is not None:
            torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, data_loader):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        self.model.eval()

        preds = []

        with torch.no_grad():

            for feature, _ in data_loader:

                feature = feature.to(self.device)

                pred = self.model(feature)

                preds.append(
                    pred.cpu()
                )

        return torch.cat(preds).numpy()

    def save(self, path):
        torch.save(
            self.model.state_dict(),
            path,
        )
    
    def load(self, path):
        state = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(state)

        self.fitted = True



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class LocalformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, d_model):
        super(LocalformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers)
        self.num_layers = num_layers

    def forward(self, src, mask):
        output = src
        out = src

        for i, mod in enumerate(self.layers):
            # [T, N, F] --> [N, T, F] --> [N, F, T]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)

            output = mod(output + out, src_mask=mask)

        return output + out


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_linear=12, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(d_feat, d_linear),
            nn.GELU(),
            nn.Linear(d_linear, d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = LocalformerEncoder(self.encoder_layer, num_layers=num_layers, d_model=d_model)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F], [512, 60, 6]
        src = self.feature_layer(src)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        mask = None

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  # [60, 512, 8]

        output, _ = self.rnn(output)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze(-1)
