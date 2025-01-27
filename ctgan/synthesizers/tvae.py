"""TVAE module."""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
from torch.nn import (
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
    BatchNorm1d,
    CrossEntropyLoss,
)
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Optional, List, Tuple, Union, Iterable
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.synthesizers.losses import MMDLoss
from ctgan.utils import monitor_loss, evaluating
from ctgan.types import SolverOptions, D, T
from tqdm import tqdm

from IPython.core.debugger import set_trace


params = {
    "vae": {
        "encoder": {"compress_dims": (128, 128), "embedding_dim": 10},
        "decoder": {"compress_dims": (128, 128), "embedding_dim": 10},
        "solver": SolverOptions(
            {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-5,
            }
        ),
    },
    "discriminator": {
        "net": {
            "hidden_dims": (15, 15),
            "add_bn": False,
        },
        "solver": SolverOptions(
            {
                "lr": 1e-4,
                "betas": (0.5, 0.9),
                "weight_decay": 1e-5,
            }
        ),
    },
}


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
        compress_dims: Iterable[int],
        embedding_dim: int,
        add_bn: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            if add_bn:
                seq += [BatchNorm1d(item)]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_: T) -> Tuple[T, T, T]:
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

    def reparameterize(self, mu: T, std: T, nsamples=1) -> T:
        if nsamples > 1:
            raise NotImplementedError
        eps = torch.randn_like(std)
        emb = eps * std + mu
        return emb


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
        decompress_dims: Iterable[int],
        data_dim: int,
        add_bn: bool = False,
    ):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            if add_bn:
                seq += [BatchNorm1d(item)]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_: T):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


class BaseTVAE(BaseSynthesizer):
    """Base TVAE class.

    Contains shared methods across TVAE variants (InfoTVAE and FactorTVAE).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        compress_dims: Iterable[int] = (128, 128),
        decompress_dims: Iterable[int] = (128, 128),
        batch_size: int = 500,
        epochs: int = 300,
        encoder_bn: bool = False,
        decoder_bn: bool = False,
        cuda: Union[str, bool] = True,
        track_loss: bool = True,
        metadata: Optional[dict] = None,  # for compatibility with SDV
    ):
        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder_bn = encoder_bn
        self.decoder_bn = decoder_bn

        self.metadata = metadata

        # logging
        self.track_loss = track_loss
        self.loss_info: dict = {}
        self.losses: List = []
        self.grads: List = []

    def _loss_function(
        self, recon_x: T, x: T, sigmas: T, mu: T, std: T, logvar: T, emb: T, output_info
    ) -> T:
        loss_rec = self._loss_rec(recon_x, x, sigmas, output_info)
        loss_reg = self._loss_reg(x, mu, logvar)
        L = loss_rec + (1 - self.alpha) * loss_reg
        if self.add_mmd:
            loss_mmd = (self.alpha + self.lbd - 1) * self._loss_mmd(
                mu, std, logvar, emb
            )
            L += loss_mmd  # mu depends of x already
        return L

    @monitor_loss(name=r"$\log \;p_\theta (x|z)$")
    def _loss_rec(self, recon_x: T, x: T, sigmas: T, output_info):
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
        return sum(loss) / x.size()[0]

    @monitor_loss(name=r"$D_{KL}(q_\phi(z|x)\,||\,p(z))$")
    def _loss_reg(self, x, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return KLD / x.size()[0]

    @monitor_loss(name=r"$D_{MMD}(q_\phi(z)\,||\,p(z))$")
    def _loss_mmd(self, mu, std, logvar, emb):
        return MMDLoss()(emb, (emb - mu) / std)

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
        with evaluating(self.encoder):
            embedded = [self.encoder(d[0].reshape(1, -1)) for d in iter(dataset)]

        mus = np.asarray([e[0].squeeze().detach().numpy() for e in embedded])
        Sigma = np.cov(mus, rowvar=False)
        return np.linalg.cond(Sigma)

    def append_losses(self, epoch, batch):
        self.losses.append(self.loss_info | {"Epoch": epoch, "Batch": batch})
        self.loss_info = {}

    def plot_losses(self, ax=None, yscale="log"):
        if ax is None:
            ax = plt.gca()
        _data = pd.DataFrame.from_dict(self.losses)
        _data = _data.loc[:, _data.columns != "Batch"]
        _data = _data.groupby(by="Epoch").mean(numeric_only=False)
        _data[r"$\mathcal{L}({\phi,\theta})$"] = _data.sum(axis=1)
        ax.set_title(self.__repr__())
        ax.set_yscale(yscale)
        ax.grid(True, which="both")
        return _data.plot(ax=ax)

    def append_gradients(self, epoch, batch):
        """Store gradients at each iteration for encoder and decoder.

        This is useful to see which term of the loss drives the adaptation.
        """

        def store_grad_magnitude(model):
            param_grads = torch.concat(
                [
                    param.grad.clone().flatten()
                    for _name, param in model.named_parameters()
                    if param.grad is not None
                    # can happen than grad(decoder.sigma) is None (no continuous feature)
                ]
            )
            return torch.linalg.norm(param_grads)

        grads = {
            "encoder": store_grad_magnitude(self.encoder),
            "decoder": store_grad_magnitude(self.decoder),
            "Epoch": epoch,
            "Batch": batch,
        }
        self.grads.append(grads)

    def plot_gradients(self, ax=None):
        raise NotImplementedError

    @property
    def discrete_columns(self):
        return [
            col
            for col, t in self.metadata.columns.items()
            if t["sdtype"] == "categorical"
        ]

    def _before_fit(self, train_data, discrete_columns, num_workers=1):
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
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        data_dim = self.transformer.output_dimensions
        self.encoder = Encoder(
            data_dim, self.compress_dims, self.embedding_dim, add_bn=self.encoder_bn
        ).to(self._device)
        self.decoder = Decoder(
            self.embedding_dim, self.decompress_dims, data_dim, add_bn=self.decoder_bn
        ).to(self._device)
        return loader

    @random_state
    def sample(self, num_rows: int, batch_size: Optional[int] = None):
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
        samples = []
        for _ in range(steps):
            mean = torch.zeros(batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake = self._decode(noise, transform=True)
            samples.append(fake)

        data = pd.concat(samples, ignore_index=True)
        data = data.iloc[:num_rows]
        return data

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
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


class TVAE(BaseTVAE):
    """InfoTVAE.

    Remains named TVAE, and default parameters give genuine TVAE for legacy.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0,
        lbd: float = 1,
        **kwargs,
    ):
        """Instantiate TVAE. Default is ELBO."""
        super().__init__(*args, **kwargs)

        assert alpha + lbd - 1 >= 0
        assert 1 >= alpha >= 0
        assert lbd > 0
        self.alpha = alpha
        self.lbd = lbd

    @property
    def add_mmd(self):
        return self.alpha + self.lbd - 1 > 0

    def __repr__(self):
        msg = "info-TVAE"
        msg += f"(alpha={self.alpha:.2f}, lambda={self.lbd:.2f}, bs={self.batch_size})"
        msg += f"\n{self.embedding_dim}-D embedding, {self.compress_dims}-D encoder, {self.decompress_dims}-D decoder"
        msg += f"\nBN: Enc({self.encoder_bn}), Dec({self.decoder_bn})"
        return msg

    @property
    def unique_id(self):
        return f"a{self.alpha:.2f}b{self.lbd:.2f}bs{self.batch_size}"

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
            **params["vae"]["solver"],
        )

        print(f"Optimization parameters: {optimizerAE}")

        self.losses = []
        for i in tqdm(range(self.epochs), desc="Epoch", position=0):
            for id_, data in tqdm(
                enumerate(loader), desc="Batch", position=1, leave=False
            ):
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
                )
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self.track_loss:
                    self.append_losses(
                        i,
                        id_,
                    )
                    # self.append_gradients(
                    #     i,
                    #     id_,
                    # )


class Discriminator(Module):
    """Discriminator MLP network.

    Is used within FactorTVAE to approximate Total Correlation with the density ratio trick.

    Outputs 2 logits and uses softmax
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: Iterable[int],
        add_bn: bool = False,
    ):
        super(Discriminator, self).__init__()
        seq = []
        dim = embedding_dim
        for item in list(hidden_dims):
            seq += [Linear(dim, item), ReLU()]
            if add_bn:
                seq += [BatchNorm1d(item)]
            dim = item

        seq.append(Linear(dim, 2))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        return self.seq(input_)


class FTVAE(TVAE):
    """FactorTVAE"""

    def __init__(
        self,
        *args,
        alpha: float = 0,
        lbd: float = 1,
        p_facto: float = 1,
        vae_batch_size: int = 64,
        facto_batch_size: int = 64,
        **kwargs,
    ) -> None:
        """Instantiate FactorTVAE."""
        super().__init__(*args, alpha=alpha, lbd=lbd, **kwargs)

        self.alpha = 0
        # assert p_reg >= 0
        if p_facto == 0:
            warnings.warn(
                "Zero weight on factorizing loss, use only for debugging purposes."
            )
        # self.p_reg = p_reg
        self.p_facto = p_facto
        self.vae_batch_size = vae_batch_size
        self.facto_batch_size = facto_batch_size
        self.batch_size = vae_batch_size + facto_batch_size

    def __repr__(self):
        msg = f"FTVAE"
        # msg += f"p_reg={self.p_reg:.2f}, "
        msg += f" ($p_{{facto}}={self.p_facto:.2f}$)"
        return msg

    @monitor_loss(name=r"$TC(z)$")
    def _loss_tc(self, logD_z):
        return (logD_z[:, 0] - logD_z[:, 1]).mean()

    def fit(self, train_data, discrete_columns: Tuple = ()) -> None:
        loader = self._before_fit(train_data, discrete_columns, num_workers=2)

        self.D = Discriminator(self.embedding_dim, **params["discriminator"]["net"]).to(
            self._device
        )

        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            **params["vae"]["solver"],
        )

        optimizerD = Adam(
            list(self.D.parameters()),
            **params["discriminator"]["solver"],
        )

        ones = torch.ones(self.facto_batch_size, dtype=torch.long, device=self._device)
        zeros = torch.zeros(
            self.facto_batch_size, dtype=torch.long, device=self._device
        )

        self.losses = []
        for i in range(self.epochs):
            for id_, batch in enumerate(loader):
                # Compute VAE loss
                optimizerAE.zero_grad()
                optimizerD.zero_grad()

                real1 = batch[0][: self.vae_batch_size].to(self._device)
                n1, _ = real1.size()
                mu1, std1, logvar1 = self.encoder(real1)
                emb1 = self.encoder.reparameterize(mu1, std1)
                rec, sigmas = self.decoder(emb1)
                lossVAE = self._loss_function(
                    rec,
                    real1,
                    sigmas,
                    mu1,
                    std1,
                    logvar1,
                    emb1,
                    self.transformer.output_info_list,
                )
                # TODO how to be sure it's log ?
                logD_z = self.D(emb1)  # returns log(D) and log(1-D)
                lossVAE = lossVAE + self.p_facto * self._loss_tc(logD_z)

                # Update VAE
                lossVAE.backward(retain_graph=True)

                optimizerAE.zero_grad()

                # Compute facto reg
                real2 = batch[0][self.vae_batch_size :].to(self._device)
                n2, _ = real2.size()
                mu2, std2, logvar2 = self.encoder(real2)
                emb2 = self.encoder.reparameterize(mu2, std2)
                z_perm = self.permute_dims(emb2).detach()
                logD_z_perm = self.D(z_perm)
                lossD = 0.5 * (
                    cross_entropy(logD_z, zeros[:n1])
                    + cross_entropy(logD_z_perm, ones[:n2])
                )
                lossD.backward()

                # Updates
                # is not really consistent with original paper
                # but update of AE parameters while retaining graph raises
                # an error since pytorch 1.5
                # TODO
                optimizerAE.step()
                optimizerD.step()

                self.decoder.sigma.data.clamp_(0.01, 1.0)

                if self.track_loss:
                    self.append_losses(i, id_)

    @staticmethod
    def permute_dims(z):
        assert z.dim() == 2

        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        return torch.cat(perm_z, 1)
