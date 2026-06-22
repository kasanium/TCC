"""
Arquitetura CNN para regressao escalar de compliancia.

Mapeia uma topologia 24x24 (com a informacao de carga) para um unico valor
de compliancia, substituindo a avaliacao por elementos finitos.

Entrada: tensor (batch, 3, 24, 24)
    canal 0: densidade da topologia
    canal 1: mapa espacial da componente Fx da carga
    canal 2: mapa espacial da componente Fy da carga
A codificacao da carga e do contorno como canais espaciais segue a abordagem
de Banga et al. (2018).

Saida: log1p(compliancia), um escalar por amostra.

A pilha convolucional (32, 64, 128 canais, nucleos 3x3, ReLU) segue a
arquitetura de Kumarn et al. (2025); o bloco denso final converte as
caracteristicas extraidas no valor escalar.

A resolucao de entrada e fixa em 24x24 (576 elementos). Para inferir em outras
malhas, e necessario redimensionar a entrada antes.
"""

import torch.nn as nn


def _softplus_fc(in_features, out_features):
    """Camada linear seguida de Softplus (ativacao suave para a regressao)."""
    return nn.Sequential(nn.Linear(in_features, out_features), nn.Softplus())


class SurrogateCNN(nn.Module):
    """CNN para regressao de compliancia em malha 24x24.

    Dimensoes ao longo da rede (entrada 24x24):
        Conv(3->32, 3x3, pad=1):    24x24 -> 24x24
        MaxPool(2x2):               24x24 -> 12x12
        Conv(32->64, 3x3, pad=1):   12x12 -> 12x12
        Conv(64->128, 3x3, pad=1):  12x12 -> 12x12
        MaxPool(2x2):               12x12 -> 6x6
        Flatten:                    128 * 6 * 6 = 4608
    """

    _FLAT_DIM = 128 * 6 * 6  # 4608

    def __init__(self):
        super().__init__()

        # extrator convolucional 32/64/128 (Kumarn et al., 2025)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # regressao escalar -> log1p(compliancia)
        self.regressor = nn.Sequential(
            _softplus_fc(self._FLAT_DIM, 256),
            _softplus_fc(256, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)