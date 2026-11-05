"""
substituto de campo para o dqn de brown 22: u-net que recebe a
topologia binaria 24x24 e as condicoes de contorno no formato do proprio
solver de brown (nos apoiados, nos carregados e sentido da carga) e preve,
por elemento, a tensao de von mises (log10) e, por uma cabeca escalar, a
energia de deformacao (log10). a saida e convertida para o mesmo formato que
FEASolve devolve (ver saida_formato_brown), para que o agente receba a mesma
observacao que receberia do mef.
diferenca para a cnn do 03: aquela preve so a compliancia, e o agente de
brown decide pelo campo de tensoes; sem o campo, a etapa 24x24 nao roda.
split treino/teste por episodio (groupshufflesplit, sem vazamento entre
estados do mesmo episodio). as trajetorias de referencia (eps = 0) dos tres
casos do benchmark ficam fora do treino e sao avaliadas a parte (so os
estados exclusivos delas: estado dividido com episodio perturbado vai para o
treino, ver 04a). com --holdout, todo estado com as condicoes de contorno de um
caso do benchmark sai do treino e da validacao.
entrada: resultados/dados/brown_tensao_24x24.npz (04a)
saidas: resultados/04b_unet_tensao.png,
resultados/dados/unet_tensao_brown_24x24.pth
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

# --- fonte casada com o corpo do texto -------------------------------------
LARGURA_FIG_POL = 13.0        # largura do figsize deste script
LARGURA_MANCHA_POL = 6.30     # 16 cm de mancha do uspsc (a4, margens 3 cm / 2 cm)
_ESC = LARGURA_FIG_POL / LARGURA_MANCHA_POL
def _pt(p): return p * _ESC   # pt desejado na pagina -> pt no matplotlib

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": _pt(12), "axes.titlesize": _pt(12), "axes.labelsize": _pt(12),
    "xtick.labelsize": _pt(12), "ytick.labelsize": _pt(12),
    "legend.fontsize": _pt(12), "figure.titlesize": _pt(12),
    "axes.linewidth": 1.5, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# constantes
# ---------------------------------------------------------------------------

EX, EY = 24, 24          # malha fina do refinamento progressivo de brown
N_NOS = (EX + 1) * (EY + 1)
SEED = 42                # semente fixa para reproduzir o split e a figura
TEST_SPLIT = 0.15        # fracao de episodios (fora os de referencia) no teste
VAL_SPLIT = 0.10         # fracao de episodios do treino usada como validacao
LAMBDA_SE = 1.0          # peso da energia de deformacao na perda
PISO_VM = 1e-4           # tensao minima, relativa ao maximo da amostra, antes do log.
                         # brown zera todo elemento abaixo de max/1e3 (FEA_SOLVER_GENERAL.py,
                         # l. 301-303), entao cortar em max/1e4 nao muda a observacao do agente
SE_MAX = 1e6             # energia acima disso (ou <= 0) = rigidez singular; estado descartado
ENERGIA_ADITIVA = True   # energia como soma de energias por elemento (ver UNetTensao)
EPOCAS = 400
PACIENCIA = 40           # epocas sem melhora na validacao antes de parar
LOTE = 128

CORES = {"U-Net": "#004488", "ideal": "#888888"}


# ---------------------------------------------------------------------------
# formato de entrada (usado tambem por 04a e 04c; manter identico)
# ---------------------------------------------------------------------------

def mapas_nos(lc_nodes, load_dirs, bc_nodes, ex=EX, ey=EY):
    """apoios e cargas de brown como mapas nodais (3, ey+1, ex+1).

    a forca e montada exatamente como em FEASolve (FEA_SOLVER_GENERAL.py,
    l. 242-246): os gdl 0..n_nos-1 sao x e n_nos..2n_nos-1 sao y, e cada par
    de nos em lc_nodes recebe o sentido de load_dirs. o modulo p/2 e omitido
    (entra como +-1). o no n fica na linha j = n // (ex+1), coluna i = n % (ex+1)
    (rectangularmesh, l. 34-38).
    """
    n_nos = (ex + 1) * (ey + 1)
    forca = np.zeros(2 * n_nos, dtype=np.float32)
    ld = 0
    for f in range(len(lc_nodes)):
        if f / 2 == int(f / 2) and f / 2 != 0:
            ld += 1
        forca[int(lc_nodes[f])] = float(load_dirs[ld])
    apoio = np.zeros(n_nos, dtype=np.float32)
    apoio[np.asarray(bc_nodes, dtype=int)] = 1.0
    return np.stack([apoio, forca[:n_nos], forca[n_nos:]]).reshape(3, ey + 1, ex + 1)


def montar_entrada(solido, mapas):
    """tensor de entrada (n, 4, ey, ex): solido, apoio, fx, fy por elemento.

    o elemento d = ex*j + i usa os nos (j,i), (j,i+1), (j+1,i+1), (j+1,i)
    (rectangularmesh, l. 39-45); o mapa nodal vira mapa de elemento pela soma
    dos quatro nos. reshape(voidcheck, (24, 24)) ja e a grade [j, i].
    """
    solido = np.asarray(solido, dtype=np.float32).reshape(-1, EY, EX)
    mapas = np.asarray(mapas, dtype=np.float32).reshape(-1, 3, EY + 1, EX + 1)
    m_el = mapas[:, :, :-1, :-1] + mapas[:, :, :-1, 1:] + mapas[:, :, 1:, :-1] + mapas[:, :, 1:, 1:]
    m_el[:, 0] = np.minimum(m_el[:, 0], 1.0)   # apoio vira indicador 0/1
    return np.concatenate([solido[:, None], m_el], axis=1).astype(np.float32)


def reconstruir_vm_nn(vm_el):
    """copia da normalizacao de brown (FEA_SOLVER_GENERAL.py, l. 296-303).

    e isso que o agente ve no canal 0: max/vm, zerado acima de 1e3 (vazios e
    elementos quase sem carga), dividido pelo maximo. vale 1 no elemento solido
    menos solicitado e 0 nos vazios. vazio previsto com tensao 0 da razao
    infinita e tambem e zerado.
    """
    v = np.asarray(vm_el, dtype=np.float64).reshape(-1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.max(v) / v
    r[~np.isfinite(r)] = 0.0
    r[r > 1e3] = 0.0
    return r / np.max(r)


# ---------------------------------------------------------------------------
# arquitetura
# ---------------------------------------------------------------------------

def _bloco(c_in, c_out):
    """duas convolucoes 3x3 com relu (bloco padrao da u-net)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.ReLU(),
    )


class UNetTensao(nn.Module):
    """u-net para o campo de tensao na malha 24x24 (ronneberger 15).

    dimensoes ao longo da rede (entrada 24x24):
        enc1 (4->32):      24x24
        enc2 (32->64):     12x12
        enc3 (64->128):    6x6
        gargalo (128->256): 3x3   (campo receptivo cobre o dominio inteiro:
                                   a tensao num ponto depende do caminho de carga todo)
        dec3, dec2, dec1 com conexoes de atalho -> 24x24
    cabeca de campo: conv 1x1 -> log10(vm) padronizado
    cabeca de energia (aditiva=True): conv 1x1 -> energia por elemento (softplus,
        zerada nos vazios), somada no dominio -> log10(energia) padronizado. a
        energia de deformacao e a soma das energias dos elementos, entao a rede
        so precisa aprender a densidade local
    cabeca de energia (aditiva=False): media global do gargalo
    """

    def __init__(self, c_in=4, aditiva=True):
        super().__init__()
        self.aditiva = aditiva
        self.enc1 = _bloco(c_in, 32)
        self.enc2 = _bloco(32, 64)
        self.enc3 = _bloco(64, 128)
        self.gargalo = _bloco(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _bloco(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _bloco(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _bloco(64, 32)
        self.saida_campo = nn.Conv2d(32, 1, kernel_size=1)
        if aditiva:
            self.saida_energia = nn.Conv2d(32, 1, kernel_size=1)
            # padronizacao do log10 da energia, gravada no proprio modelo
            self.register_buffer("media_se", torch.tensor(0.0))
            self.register_buffer("desv_se", torch.tensor(1.0))
        else:
            self.saida_energia = nn.Sequential(nn.Linear(256, 64), nn.Softplus(), nn.Linear(64, 1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        g = self.gargalo(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(g), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        campo = self.saida_campo(d1)[:, 0]
        if self.aditiva:
            w = F.softplus(self.saida_energia(d1)[:, 0]) * x[:, 0]   # so solidos
            log_se = torch.log10(w.sum(dim=(1, 2)) + 1e-12)
            energia = (log_se - self.media_se) / self.desv_se
        else:
            energia = self.saida_energia(g.mean(dim=(2, 3)))[:, 0]
        return campo, energia


# ---------------------------------------------------------------------------
# uso do modelo treinado (04c importa estas funcoes)
# ---------------------------------------------------------------------------

def carregar_unet(caminho):
    """carrega pesos e constantes de padronizacao salvos por este script."""
    ck = torch.load(caminho, map_location="cpu")
    modelo = UNetTensao(aditiva=ck.get("info", {}).get("aditiva", False))
    modelo.load_state_dict(ck["estado"])
    modelo.eval()
    return modelo, ck["norm"]


def prever_brown(modelo, norm, solido, mapas):
    """preve (energia, vm_el) para um estado; vm_el tem 0 nos vazios."""
    x = torch.from_numpy(montar_entrada(solido, mapas))
    with torch.no_grad():
        pc, ps = modelo(x)
    log_vm = pc[0].numpy() * norm["desv_vm"] + norm["media_vm"]
    log_se = float(ps[0]) * norm["desv_se"] + norm["media_se"]
    vm_el = (10.0 ** log_vm).ravel() * np.asarray(solido, dtype=np.float64).ravel()
    return 10.0 ** log_se, vm_el


def saida_formato_brown(se, vm_el, ex=EX, ey=EY):
    """tupla no formato de FEASolve: (desloc., energia, vm_el, vm_nn, vm_nos).

    indices usados pelo ambiente de brown: [1] energia (max_se_tot, max_se_ep),
    [2] vm por elemento (so na norma-p da restricao de tensao, inativa com
    sc = 10), [3] vm normalizado (observacao do agente). [0] e [4] nao sao lidos
    por nenhum trecho do ambiente; voltam como placeholder.
    """
    vm_el = np.asarray(vm_el, dtype=np.float64)
    return (0.0, np.array([[se]]), vm_el.reshape(ex, ey), reconstruir_vm_nn(vm_el),
            np.ones(((ex + 1) * (ey + 1), 1)))


# ---------------------------------------------------------------------------
# utilitarios
# ---------------------------------------------------------------------------

def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def alvos_log(vm_el, se, solido):
    """log10 da tensao (com piso relativo) e da energia; mascara de solidos."""
    vm = np.asarray(vm_el, dtype=np.float64)
    piso = PISO_VM * vm.max(axis=1, keepdims=True)
    log_vm = np.log10(np.maximum(vm, piso)).astype(np.float32).reshape(-1, EY, EX)
    mask = np.asarray(solido, dtype=np.float32).reshape(-1, EY, EX)
    log_se = np.log10(np.asarray(se, dtype=np.float64)).astype(np.float32)
    return log_vm, mask, log_se


def _espelha(xb, cb, mb):
    """aumento de dados por simetria: espelhar em x inverte fx, em y inverte fy.

    a tensao de von mises e a energia nao mudam com o espelhamento.
    canais da entrada: 0 solido, 1 apoio, 2 fx, 3 fy.
    """
    if torch.rand(1).item() < 0.5:
        xb, cb, mb = xb.flip(-1), cb.flip(-1), mb.flip(-1)
        xb[:, 2] = -xb[:, 2]
    if torch.rand(1).item() < 0.5:
        xb, cb, mb = xb.flip(-2), cb.flip(-2), mb.flip(-2)
        xb[:, 3] = -xb[:, 3]
    return xb, cb, mb


def _perda(pc, ps, cb, mb, sb):
    """mse do log da tensao so nos solidos + mse do log da energia."""
    campo = ((pc - cb) ** 2 * mb).sum() / mb.sum().clamp(min=1.0)
    return campo + LAMBDA_SE * ((ps - sb) ** 2).mean()


def _prever_lotes(modelo, x, lote=512):
    dev = next(modelo.parameters()).device
    pcs, pss = [], []
    with torch.no_grad():
        for i in range(0, len(x), lote):
            pc, ps = modelo(x[i:i + lote].to(dev))
            pcs.append(pc.cpu())
            pss.append(ps.cpu())
    return torch.cat(pcs), torch.cat(pss)


# ---------------------------------------------------------------------------
# treino
# ---------------------------------------------------------------------------

def treinar_unet(xt, ct, mt, st, xv, cv, mv, sv, norm, epocas=EPOCAS):
    """treina a u-net com adam, reducao de lr e parada antecipada.

    a validacao e um grupo de episodios separado do treino (sem tocar no
    teste); interrompe se a perda de validacao nao melhora por PACIENCIA epocas.
    treina na gpu se houver (a inferencia no laco do agente e em cpu).
    """
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = UNetTensao(aditiva=ENERGIA_ADITIVA)
    if ENERGIA_ADITIVA:
        modelo.media_se.fill_(norm["media_se"])
        modelo.desv_se.fill_(norm["desv_se"])
    modelo.to(dev)
    print(f"  dispositivo de treino: {dev}")
    opt = optim.Adam(modelo.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loader = DataLoader(TensorDataset(xt, ct, mt, st), batch_size=LOTE, shuffle=True)
    melhor_v, pac, melhor_w = float("inf"), 0, None
    hist_t, hist_v = [], []
    for ep in range(epocas):
        modelo.train()
        soma, n = 0.0, 0
        for xb, cb, mb, sb in loader:
            xb, cb, mb, sb = xb.to(dev), cb.to(dev), mb.to(dev), sb.to(dev)
            xb, cb, mb = _espelha(xb, cb, mb)
            opt.zero_grad()
            pc, ps = modelo(xb)
            perda = _perda(pc, ps, cb, mb, sb)
            perda.backward()
            opt.step()
            soma += perda.item() * len(xb)
            n += len(xb)
        modelo.eval()
        pc, ps = _prever_lotes(modelo, xv)
        v = _perda(pc, ps, cv, mv, sv).item()
        sched.step(v)
        hist_t.append(soma / n)
        hist_v.append(v)
        if v < melhor_v:
            melhor_v = v
            melhor_w = {k: w.clone() for k, w in modelo.state_dict().items()}
            pac = 0
        else:
            pac += 1
        if (ep + 1) % 10 == 0:
            print(f"  epoca {ep+1:3d} | treino {hist_t[-1]:.4f} | validacao {v:.4f}")
        if pac >= PACIENCIA:
            print(f"  parada antecipada na epoca {ep+1}")
            break
    modelo.load_state_dict(melhor_w)
    return modelo.to("cpu"), hist_t, hist_v


# ---------------------------------------------------------------------------
# avaliacao
# ---------------------------------------------------------------------------

def avaliar(modelo, norm, x, solido, vm_el, vm_nn, se):
    """metricas no espaco que importa para o agente.

    r2_se:     log10 da energia de deformacao
    r2_logvm:  log10 da tensao nos solidos
    r2_vmnn:   vm normalizado (observacao do agente) nos solidos
    spearman:  correlacao de postos do vm normalizado, media por amostra
               (o agente escolhe por comparacao entre elementos)
    top1_top5: fracao das amostras em que o elemento que o substituto aponta
               como menos solicitado (max de vm_nn) esta entre os 5 menos
               solicitados segundo o mef
    """
    pc, ps = _prever_lotes(modelo, torch.from_numpy(x))
    log_vm_p = pc.numpy() * norm["desv_vm"] + norm["media_vm"]
    log_se_p = ps.numpy() * norm["desv_se"] + norm["media_se"]
    sol = np.asarray(solido, dtype=bool).reshape(len(x), -1)

    vm_p = (10.0 ** log_vm_p).reshape(len(x), -1) * sol
    nn_p = np.stack([reconstruir_vm_nn(v).ravel() for v in vm_p])
    nn_r = np.asarray(vm_nn, dtype=np.float64).reshape(len(x), -1)

    log_vm_r, _, log_se_r = alvos_log(vm_el, se, solido)
    log_vm_r = log_vm_r.reshape(len(x), -1)

    rho, acerto = [], []
    for i in range(len(x)):
        s = sol[i]
        if s.sum() > 5:
            rho.append(spearmanr(nn_p[i, s], nn_r[i, s]).correlation)
        a = int(np.argmax(nn_p[i]))
        acerto.append(nn_r[i, a] >= np.sort(nn_r[i])[-5])

    return {
        "r2_se": r2_score(log_se_r, log_se_p),
        "r2_logvm": r2_score(log_vm_r[sol], log_vm_p.reshape(len(x), -1)[sol]),
        "r2_vmnn": r2_score(nn_r[sol], nn_p[sol]),
        "spearman": float(np.nanmean(rho)),
        "top1_top5": float(np.mean(acerto)),
        "_log_se": (log_se_r, log_se_p),
        "_log_vm": (log_vm_r[sol], log_vm_p.reshape(len(x), -1)[sol]),
        "_vmnn": (nn_r, nn_p),
    }


def _imprime(nome, m):
    print(f"  {nome:<26} | R2 log SE {m['r2_se']:.4f} | R2 log VM {m['r2_logvm']:.4f} | "
          f"R2 VM_nn {m['r2_vmnn']:.4f} | Spearman {m['spearman']:.3f} | "
          f"top1-em-top5 {m['top1_top5']:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="tira do treino e da validacao todos os estados com a combinacao de "
                         "apoio/carga de um caso do benchmark (e todo estado dos episodios "
                         "do benchmark); eles vao so para o teste")
    args = ap.parse_args()
    suf = "_holdout" if args.holdout else ""
    _HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_HERE, "resultados", "dados", "brown_tensao_24x24.npz")
    out_model = os.path.join(_HERE, "resultados", "dados", f"unet_tensao_brown_24x24{suf}.pth")
    out_fig = os.path.join(_HERE, "resultados", f"04b_unet_tensao{suf}.png")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    if not os.path.exists(data_path):
        print("erro: dataset nao encontrado. rode 04a_dados_brown.py antes.")
        sys.exit(1)

    d = np.load(data_path)
    # alguns casos sorteados deixam a estrutura sem apoio suficiente: a matriz de
    # rigidez fica singular e o solver devolve energia negativa ou acima de 1e10.
    # entre 1e6 e 1e10 nao ha nenhum estado, entao o corte em 1e6 separa os dois grupos
    valido = (d["se"] > 0) & (d["se"] < SE_MAX)
    solido = d["solido"][valido].astype(np.float32)
    mapas = d["mapas"][d["mapa_idx"][valido]]
    vm_el, vm_nn, se = d["vm_el"][valido], d["vm_nn"][valido], d["se"][valido]
    episodio, caso, ref = d["episodio"][valido], d["caso"][valido], d["ref"][valido].astype(bool)
    mapa_idx = d["mapa_idx"][valido]
    print(f"dataset: {len(solido)} estados | {len(np.unique(episodio))} episodios | "
          f"{len(d['mapas'])} combinacoes de apoio/carga | "
          f"{int((~valido).sum())} estados singulares descartados")

    # sanidade: a normalizacao copiada de brown tem de reproduzir o canal real
    diff = max(np.abs(reconstruir_vm_nn(vm_el[i]).ravel() - vm_nn[i]).max()
               for i in range(0, len(vm_el), max(1, len(vm_el) // 200)))
    print(f"sanidade reconstruir_vm_nn x FEASolve[3]: max |dif| = {diff:.2e} (esperado ~1e-6)")
    if diff > 1e-3:
        print("erro: a normalizacao nao bate com a de brown; nao siga para o 04c.")
        sys.exit(1)

    X = montar_entrada(solido, mapas)
    log_vm, mask, log_se = alvos_log(vm_el, se, solido)

    # split: referencia (eps = 0 nos tres casos do benchmark) fica toda no teste;
    # o resto e dividido por episodio
    idx_ref = np.where(ref)[0]
    idx_rest = np.where(~ref)[0]
    idx_bench = np.array([], dtype=int)
    if args.holdout:
        # combinacao (BC_Nodes, LC_Nodes, Load_Directions) de cada caso do benchmark =
        # mapa mais frequente nos episodios do caso (os outros mapas desses episodios
        # vem da 1a chamada, com apoios sorteados). o 04a deduplica os mapas byte a
        # byte, entao combinacao igual = mesmo indice, inclusive em caso sorteado
        bench_mapas = [int(np.bincount(mapa_idx[caso == c]).argmax()) for c in range(3)]
        por_mapa = np.isin(mapa_idx, bench_mapas)
        por_episodio = caso < 3
        fora = por_mapa | por_episodio
        coincid = por_mapa & ~por_episodio
        print(f"holdout: {int(fora.sum())} estados fora do treino e da validacao "
              f"({int(por_mapa.sum())} com a combinacao de apoio/carga do benchmark, dos quais "
              f"{int(coincid.sum())} de casos sorteados que coincidem; "
              f"{int((por_episodio & ~por_mapa).sum())} das 1as chamadas, com apoios sorteados, "
              f"dos episodios do benchmark)")
        idx_bench = np.where(por_mapa)[0]
        idx_ref = np.where(ref & por_mapa)[0]
        idx_rest = np.where(~fora)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
    tr, te = next(gss.split(idx_rest, groups=episodio[idx_rest]))
    idx_tr, idx_te = idx_rest[tr], idx_rest[te]
    gss_v = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    tr2, va = next(gss_v.split(idx_tr, groups=episodio[idx_tr]))
    idx_tr, idx_va = idx_tr[tr2], idx_tr[va]
    print(f"split: treino {len(idx_tr)} | validacao {len(idx_va)} | teste {len(idx_te)} | "
          f"referencia (fora do treino) {len(idx_ref)}")

    # padronizacao com estatisticas do treino
    norm = {
        "media_vm": float(log_vm[idx_tr][mask[idx_tr] > 0].mean()),
        "desv_vm": float(log_vm[idx_tr][mask[idx_tr] > 0].std()),
        "media_se": float(log_se[idx_tr].mean()),
        "desv_se": float(log_se[idx_tr].std()),
    }
    c_std = (log_vm - norm["media_vm"]) / norm["desv_vm"]
    s_std = (log_se - norm["media_se"]) / norm["desv_se"]

    def _t(idx):
        return (torch.from_numpy(X[idx]), torch.from_numpy(c_std[idx]),
                torch.from_numpy(mask[idx]), torch.from_numpy(s_std[idx]))

    print("\ntreinando u-net...")
    t0 = time.time()
    modelo, hist_t, hist_v = treinar_unet(*_t(idx_tr), *_t(idx_va), norm)
    print(f"treino concluido em {time.time() - t0:.0f} s")

    torch.save({"estado": modelo.state_dict(), "norm": norm,
                "info": {"n_treino": int(len(idx_tr)), "seed": SEED,
                         "aditiva": ENERGIA_ADITIVA, "holdout": bool(args.holdout)}}, out_model)
    print(f"salvo: {out_model}")

    # --- metricas ---------------------------------------------------------
    print("\nmetricas:")
    m_te = avaliar(modelo, norm, X[idx_te], solido[idx_te], vm_el[idx_te], vm_nn[idx_te], se[idx_te])
    _imprime("teste (episodios novos)", m_te)
    if len(idx_bench):
        _imprime(f"benchmark holdout ({len(idx_bench)})",
                 avaliar(modelo, norm, X[idx_bench], solido[idx_bench], vm_el[idx_bench],
                         vm_nn[idx_bench], se[idx_bench]))
        for c, nome in enumerate(["cantilever", "ponte", "console"]):
            ic = idx_bench[mapa_idx[idx_bench] == bench_mapas[c]]
            if len(ic) > 1:
                _imprime(f"  holdout {nome} ({len(ic)})",
                         avaliar(modelo, norm, X[ic], solido[ic], vm_el[ic], vm_nn[ic], se[ic]))
    m_ref = None
    if len(idx_ref):
        m_ref = avaliar(modelo, norm, X[idx_ref], solido[idx_ref], vm_el[idx_ref],
                        vm_nn[idx_ref], se[idx_ref])
        _imprime("referencia (3 casos)", m_ref)
        for c, nome in enumerate(["cantilever", "ponte", "console"]):
            ic = idx_ref[caso[idx_ref] == c]
            if len(ic) > 1:
                _imprime(f"  ref. {nome} ({len(ic)})",
                         avaliar(modelo, norm, X[ic], solido[ic], vm_el[ic], vm_nn[ic], se[ic]))

    # latencia de uma previsao isolada (como no laco do agente)
    torch.set_num_threads(1)
    i0 = int(idx_te[0])
    mp = mapas[i0]
    for _ in range(5):
        prever_brown(modelo, norm, solido[i0], mp)
    t0 = time.perf_counter()
    for _ in range(200):
        saida_formato_brown(*prever_brown(modelo, norm, solido[i0], mp))
    lat = (time.perf_counter() - t0) / 200
    print(f"\nlatencia por avaliacao (1 thread, inclui montar entrada e normalizar): {lat*1e3:.2f} ms")

    m_alvo = m_ref if m_ref is not None else m_te
    if m_alvo["top1_top5"] < 0.8 or m_alvo["spearman"] < 0.9:
        print("aviso: com esse erro o agente provavelmente vai escolher elementos diferentes "
              "do que escolheria com o mef. rode o 04c mesmo assim e compare as topologias.")

    # --- figura -------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.6))

    ax = axs[0, 0]
    r, p = m_te["_log_se"]
    ax.scatter(r, p, s=6, alpha=0.4, color=CORES["U-Net"], edgecolors="none")
    lim = [min(r.min(), p.min()), max(r.max(), p.max())]
    ax.plot(lim, lim, "--", color=CORES["ideal"], lw=1.5)
    ax.set_xlabel(r"$\log_{10}$ energia (MEF)")
    ax.set_ylabel(r"$\log_{10}$ energia (U-Net)")
    ax.set_title(f"$R^2 = {m_te['r2_se']:.3f}$".replace(".", "{,}"))
    _spines(ax)

    ax = axs[0, 1]
    r, p = m_te["_log_vm"]
    sel = np.random.default_rng(SEED).choice(len(r), size=min(20000, len(r)), replace=False)
    ax.hexbin(r[sel], p[sel], gridsize=50, cmap="Blues", mincnt=1, bins="log")
    lim = [r.min(), r.max()]
    ax.plot(lim, lim, "--", color=CORES["ideal"], lw=1.5)
    ax.set_xlabel(r"$\log_{10}\sigma_{VM}$ (MEF)")
    ax.set_ylabel(r"$\log_{10}\sigma_{VM}$ (U-Net)")
    ax.set_title(f"$R^2 = {m_te['r2_logvm']:.3f}$".replace(".", "{,}"))
    _spines(ax)

    ax = axs[0, 2]
    ax.plot(hist_t, color=CORES["U-Net"], lw=1.5, label="treino")
    ax.plot(hist_v, color=CORES["U-Net"], lw=1.5, ls="--", label="validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda")
    ax.legend(frameon=False, loc="upper right")
    _spines(ax)

    # exemplo: ultimo estado da trajetoria de referencia do cantilever (ou do teste)
    if len(idx_ref) and np.any(caso[idx_ref] == 0):
        iex = int(idx_ref[caso[idx_ref] == 0][-1])
    else:
        iex = int(idx_te[-1])
    _, vm_p = prever_brown(modelo, norm, solido[iex], mapas[iex])
    nn_p = reconstruir_vm_nn(vm_p).reshape(EY, EX)
    nn_r = vm_nn[iex].reshape(EY, EX)
    paineis = [(nn_r, "Observação (MEF)", "viridis", 0, 1),
               (nn_p, "Observação (U-Net)", "viridis", 0, 1),
               (np.abs(nn_p - nn_r), "|Diferença|", "Reds", 0, None)]
    for col, (img, tit, cmap, vmin, vmax) in enumerate(paineis):
        ax = axs[1, col]
        im = ax.imshow(img, cmap=cmap, origin="upper", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        ax.set_title(tit)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _spines(ax)

    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"salvo: {out_fig}")
