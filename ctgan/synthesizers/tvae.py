"""TVAE module."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, BatchNorm1d
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, List
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
        add_bn (bool):
            Whether to add batch normalization.
    """

    def __init__(
        self,
        data_dim: int,
        compress_dims: List[int],
        embedding_dim: List[int],
        add_bn: bool = False,
    ):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            if add_bn:
                seq += BatchNorm1d(item)
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`.

        Args:
            input_ (Tensor):
                Batch of data

        Returns:
            mu (Tensor)
            std (Tensor)
            logvar (Tensor)
        """
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar

    def reparameterize(self, mu, std, nsamples=1):
        if nsamples > 1:
            raise NotImplementedError
        eps = torch.randn_like(std)
        emb = eps * std + mu
        return emb

    def mutual_info_q(self, input_):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Implemented from https://github.com/jxhe/vae-lagging-encoder

        Returns: Float
        """
        mu, std, logvar = self.forward(input_)
        emb = self.reparameterize(mu, std)
        return self._mi_from_mu(mu, std, logvar, emb)

    def _mi_from_mu(self, mu, std, logvar, z):
        """See `self.mutual_info_q`."""
        batch_size, embedding_dim = mu.size()

        log2pi = np.log(2 * np.pi)

        neg_entropy = -0.5 * embedding_dim * log2pi - 0.5 * (1 + logvar).sum(-1).mean()

        dev = z - mu

        log_density = (
            -0.5 * ((dev / std) ** 2).sum(dim=-1)
            - 0.5 * (embedding_dim * log2pi)
            + logvar.sum(-1)
        )

        log_qz = torch.logsumexp(log_density, dim=-1) - np.log(batch_size)

        return neg_entropy - log_qz.mean(-1)


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(
        self,
        embedding_dim: int,
        decompress_dims: List[int],
        data_dim: List[int],
        add_bn: bool = False,
    ):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            if add_bn:
                seq += BatchNorm1d()
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def gaussian_kernel(a, b, ell=0.5):
    """Gaussian kernel between vectors a and b

    Args:
        a (Tensor): n x d
        b (Tensor): n x d

    Returns:
        _type_: _description_
    """
    denom = 2 * ell**2
    num = (a - b).pow(2).mean(2)
    return torch.exp(-num / denom)


class TVAE(BaseSynthesizer):
    """(info-)TVAE."""

    def _loss_function(
        self, recon_x, x, sigmas, mu, std, logvar, emb, output_info, factor, epoch
    ):
        loss_rec, loss_reg = self._loss_elbo(
            recon_x, x, sigmas, mu, logvar, output_info, factor
        )
        loss_reg = (1 - self.alpha) * loss_reg
        L = loss_rec + loss_reg
        if self.add_mmd:
            loss_mmd = (self.alpha + self.lbd - 1) * self._loss_mmd(
                mu, std, logvar, emb
            )
            L += loss_mmd  # mu depends of x already
        if self.add_cond:
            raise NotImplementedError
            # loss += kappa
        if self.track_loss:
            loss_info = {
                r"$\log \;p_\theta (x|z)$": loss_rec.detach().cpu().numpy(),
                r"$(1-\alpha)D_{KL}(q_\phi(z|x)\,||\,p(z))$": loss_reg.detach()
                .cpu()
                .numpy(),
                "Epoch": epoch,
            }
            if self.add_mmd:
                loss_info[r"$(\alpha + \lambda - 1)D_{MMD}(q_\phi(z)\,||\,p(z))$"] = (
                    loss_mmd.detach().cpu().numpy()
                )
            if self.add_cond:
                loss_info["Cond"] = kappa.detach().cpu().numpy()
            self.append_losses(loss_info)
        return L

    @staticmethod
    def _loss_elbo(recon_x, x, sigmas, mu, logvar, output_info, factor):
        # reconstruction error
        st = 0
        loss = []
        for column_info in output_info:
            for span_info in column_info:
                if span_info.activation_fn != "softmax":
                    ed = st + span_info.dim
                    std = sigmas[st]
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                    loss.append((eq**2 / 2 / (std**2)).sum())
                    loss.append(torch.log(std) * x.size()[0])
                    st = ed

                else:
                    ed = st + span_info.dim
                    loss.append(
                        cross_entropy(
                            recon_x[:, st:ed],
                            torch.argmax(x[:, st:ed], dim=-1),
                            reduction="sum",
                        )
                    )
                    st = ed
        assert st == recon_x.size()[1]
        # KLD regularization
        epsilon = 0.5
        KLD = -0.5 * torch.sum(
            1
            + logvar
            - (mu / 1) ** 2
            - logvar.exp()
            # + torch.log((mu - epsilon) ** 2)
        )
        return (sum(loss) * factor / x.size()[0], KLD / x.size()[0])

    def _loss_cond(mu):
        cov = torch.cov(mu.T)
        # conditioning penalization
        # kappa = torch.logsumexp(sigmas, dim=-1)
        # kappa /= 1 / torch.logsumexp(1 / sigmas, dim=-1)
        # kappa -= 1

    def _loss_mmd(self, mu, std, logvar, emb):
        return MMDLoss()(emb, (emb - mu) / std)

    def __init__(
        self,
        embedding_dim: int = 128,
        compress_dims: npt.ArrayLike = (128, 128),
        decompress_dims: npt.ArrayLike = (128, 128),
        l2scale: float = 1e-5,
        alpha: float = 0,
        lbd: float = 1,
        batch_size: int = 500,
        epochs: int = 300,
        loss_factor: int = 2,
        add_cond: bool = False,
        add_mmd: bool = True,
        cuda: bool = True,
        metadata: bool = None,  # for compatibility with SDV
        track_loss: bool = True,
    ):
        """Instantiate TVAE. Default is ELBO."""
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        assert alpha + lbd - 1 >= 0
        assert 1 >= alpha >= 0
        assert lbd > 0
        self.metadata = metadata
        self.alpha = alpha
        self.lbd = lbd
        self.add_cond = add_cond
        self.add_mmd = add_mmd
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        # logging
        self.track_loss = track_loss
        self.losses = []
        self.lrs = []

    def __repr__(self):
        msg = "info-TVAE"
        msg += f"(alpha={self.alpha:.2f}, lambda={self.lbd:.2f}, bs={self.batch_size})"  # add_cond={self.add_cond}, add_mmd={self.add_mmd},
        msg += f"\n{self.embedding_dim}-D embedding, {self.compress_dims}-D encoder, {self.decompress_dims}-D decoder"
        return msg

    @property
    def unique_id(self):
        return f"b{self.lbd}r{int(self.add_cond)}bs{self.batch_size}"

    def append_losses(self, loss_info):
        self.losses.append(loss_info)

    def plot_losses(self, ax=None):
        if ax is None:
            ax = plt.gca()
        _data = pd.DataFrame.from_dict(self.losses)
        _data = _data.groupby(by="Epoch").mean()
        _data[r"$\mathcal{L}({\phi,\theta})$"] = _data.sum(axis=1)
        return _data.plot(ax=ax)

    @property
    def discrete_columns(self):
        return [
            col
            for col, t in self.metadata.columns.items()
            if t["sdtype"] == "categorical"
        ]

    def _before_fit(self, train_data, discrete_columns):
        if self.metadata is not None:
            warnings.warn("Using `metadata` to determine discrete columns.")
            discrete_columns = self.discrete_columns

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(
            torch.from_numpy(train_data.astype("float32")).to(self._device)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        data_dim = self.transformer.output_dimensions
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(
            self._device
        )
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(
            self._device
        )
        return loader

    def compute_cond(self, data: pd.DataFrame):
        """Compute condition number of data in latent space.

        Args:
            data (pd.DataFrame): Similar to train data.
        Returns: Float
        """
        data = self.transformer.transform(data)
        dataset = TensorDataset(
            torch.from_numpy(data.astype("float32")).to(self._device)
        )
        embedded = [self.encoder(d[0]) for d in dataset]
        mus = np.asarray([e[0].detach().numpy() for e in embedded])
        # sigmas = np.asarray([e[1].detach().numpy() for e in embedded])
        Sigma = np.cov(mus, rowvar=False)
        return np.linalg.cond(Sigma)

    def adapt_beta(self):
        raise NotImplementedError

    def _gradient_norm(self):
        # https://discuss.pytorch.org/t/check-the-norm-of-gradients/27961/5
        raise NotImplementedError

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        loader = self._before_fit(train_data, discrete_columns)

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale,
            lr=1e-3,
        )
        # schedulerAE = ReduceLROnPlateau(optimizerAE, "min")
        # schedulerAE = StepLR(optimizerAE, step_size=50, gamma=0.5, verbose=True)

        print(f"Optimization parameters: {optimizerAE}")

        self.losses = []
        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                emb = self.encoder.reparameterize(mu, std)
                rec, sigmas = self.decoder(emb)
                loss = self._loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    std,
                    logvar,
                    emb,
                    self.transformer.output_info_list,
                    self.loss_factor,
                    i,
                )
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
            # lmmd = self._loss_mmd()
            # schedulerAE.step()

    # def calc_mi(self, data: DataLoader):
    #     mi = 0
    #     num_examples = 0
    #     for batch_data in data:
    #         batch_data = batch_data[0]
    #         batch_size = batch_data.size(0)
    #         num_examples += batch_size
    #         mutual_info = self.encoder.mutual_info_q(batch_data)
    #         mi += mutual_info * batch_size
    #     return mi / num_examples

    def fit_aggressive(self, train_data, discrete_columns, eps_mi=1e-3):
        # TODO
        loader = self._before_fit(train_data, discrete_columns)

        optimizerEnc = Adam(
            self.encoder.parameters(),
            weight_decay=self.l2scale,
        )
        optimizerDec = Adam(
            self.decoder.parameters(),
            weight_decay=self.l2scale,
        )
        aggressive_flag = True
        for i in range(self.epochs):
            i_aggressive = 0
            while aggressive_flag:
                mi = self.calc_mi(loader)
                for id_, data in enumerate(loader):
                    optimizerEnc.zero_grad()
                    real = data[0].to(self._device)
                    mu, std, logvar = self.encoder(real)
                    emb = self.encoder.reparameterize(mu, std)
                    rec, sigmas = self.decoder(emb)
                    loss = self._loss_function(
                        rec,
                        real,
                        sigmas,
                        mu,
                        std,
                        logvar,
                        emb,
                        self.transformer.output_info_list,
                        self.loss_factor,
                        i,
                    )
                    loss.backward()
                    optimizerEnc.step()
                # count iters and exit if > 10
                i_aggressive += 1
                if i_aggressive > 10:
                    aggressive_flag = False
                # compute mi and exit if no improvement
                mi_prev = mi
                mi = self.calc_mi(loader)
                if mi - mi_prev < eps_mi:
                    aggressive_flag = False

            for id_, data in enumerate(loader):
                optimizerEnc.zero_grad()
                optimizerDec.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                emb = self.encoder.reparameterize(mu, std)
                rec, sigmas = self.decoder(emb)
                loss = self._loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    std,
                    emb,
                    self.transformer.output_info_list,
                    self.loss_factor,
                    i,
                )
                loss.backward()
                optimizerEnc.step()
                optimizerDec.step()

    @random_state
    def sample(self, num_rows, batch_size: int = None):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        if batch_size is None:
            batch_size = self.batch_size

        steps = num_rows // batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake = self._decode(noise, transform=True)
            data.append(fake)

        data = pd.concat(data, ignore_index=True)
        data = data.iloc[:num_rows]
        return data

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)

    def _decode(self, noise, transform=False):
        fake, sigmas = self.decoder(noise)
        fake = torch.tanh(fake)
        if transform:
            rec = self.transformer.inverse_transform(
                fake.detach().cpu().numpy(), sigmas.detach().cpu().numpy()
            )
        else:
            rec = fake
        return rec


import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
