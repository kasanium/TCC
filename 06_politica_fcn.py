"""
politica totalmente convolucional para o maskableppo: o tronco preserva a
resolucao espacial e a cabeca do ator emite um logit por elemento via
convolucao 1x1 — sem pooling e sem camada densa antes da saida, o padrao de
brown et al. (2022, fig. 6, quatro convolucoes 16/8/4/1) e de kumarn et al.
(2025). motivo: com adaptiveavgpool antes das densas nao existe caminho da
tensao do elemento i ate o logit i, e a politica gulosa eso (que empata com
o simp) nem pertence a classe de funcoes representaveis. o critico usa
pooling global + mlp (pode ser global; so o ator precisa de resolucao).
action_net = identity, com o encolhimento de ganho 0.01 aplicado a mao na
ultima convolucao.

os checkpoints salvos referenciam este modulo pelo nome "politica_fcn";
carregar sempre registrando esse nome em sys.modules (ver 07/08/09/11).
"""

import sys

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# os .zip do sb3 desserializam a politica pelo nome de modulo gravado no
# checkpoint ("politica_fcn"). quem carrega este arquivo por caminho deve
# registrar o modulo em sys.modules["politica_fcn"] ANTES do exec_module
# (padrao usado em 07, 08, 09 e 11); a linha abaixo cobre o import normal
if __name__ in sys.modules:
    sys.modules.setdefault("politica_fcn", sys.modules[__name__])


class TopoConvTrunk(BaseFeaturesExtractor):
    """Tronco convolucional que PRESERVA a resolucao espacial.

    As features devolvidas sao o mapa (C, H, W) achatado; o mlp_extractor o
    remonta. O achatamento existe so porque a API do SB3 exige um vetor.
    """

    def __init__(self, obs_space: gym.spaces.Box, n_filters=(32, 64, 64)):
        c_in, h, w = obs_space.shape
        super().__init__(obs_space, features_dim=n_filters[-1] * h * w)
        self.h, self.w = h, w
        self.c_out = n_filters[-1]

        camadas, c = [], c_in
        for f in n_filters:
            camadas += [nn.Conv2d(c, f, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()]
            c = f
        self.cnn = nn.Sequential(*camadas)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs).flatten(1)


class TopoFCNExtractor(nn.Module):
    """Substitui o mlp_extractor do SB3.

    latent_pi tem dimensao H*W e JA SAO os logits (a action_net vira
    Identity na politica). latent_vf e um vetor global, como de costume.
    """

    def __init__(self, c_out: int, h: int, w: int, vf_hidden: int = 128):
        super().__init__()
        self.c_out, self.h, self.w = c_out, h, w
        self.latent_dim_pi = h * w
        self.latent_dim_vf = vf_hidden

        # cabeca do ator: 1x1 -> um logit por elemento
        self.pi_head = nn.Sequential(
            nn.Conv2d(c_out, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        # cabeca do critico: pode ser global
        self.vf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c_out, vf_hidden), nn.ReLU(),
            nn.Linear(vf_hidden, vf_hidden), nn.ReLU(),
        )

    def _mapa(self, features: torch.Tensor) -> torch.Tensor:
        return features.view(-1, self.c_out, self.h, self.w)

    def forward(self, features):
        m = self._mapa(features)
        return self.pi_head(m).flatten(1), self.vf_head(m)

    def forward_actor(self, features):
        return self.pi_head(self._mapa(features)).flatten(1)

    def forward_critic(self, features):
        return self.vf_head(self._mapa(features))


class TopoFCNPolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy com cabeca de ator totalmente convolucional."""

    def __init__(self, *args, nelx: int = 30, nely: int = 30,
                 n_filters=(32, 64, 64), vf_hidden: int = 128, **kwargs):
        self._nelx, self._nely = nelx, nely
        self._n_filters, self._vf_hidden = n_filters, vf_hidden
        kwargs["features_extractor_class"] = TopoConvTrunk
        kwargs["features_extractor_kwargs"] = dict(n_filters=n_filters)
        kwargs["net_arch"] = []          # ignorado; o extrator abaixo manda
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TopoFCNExtractor(
            c_out=self._n_filters[-1], h=self._nely, w=self._nelx,
            vf_hidden=self._vf_hidden)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # os logits ja saem da cabeca convolucional; nada de camada densa
        # de 64 -> nelx*nely no caminho
        self.action_net = nn.Identity()
        # o SB3 inicializa a action_net com ganho 0.01 para a politica comecar
        # quase uniforme; como a saida agora vem de dentro do mlp_extractor
        # (ganho sqrt(2)), o mesmo encolhimento e aplicado a mao na ultima
        # convolucao da cabeca do ator
        if self.ortho_init:
            with torch.no_grad():
                ultima = self.mlp_extractor.pi_head[-1]
                ultima.weight.mul_(0.01)
                if ultima.bias is not None:
                    ultima.bias.zero_()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def fcn_policy_kwargs(nelx: int, nely: int, n_filters=(32, 64, 64),
                      vf_hidden: int = 128) -> dict:
    return dict(nelx=nelx, nely=nely, n_filters=n_filters, vf_hidden=vf_hidden)
