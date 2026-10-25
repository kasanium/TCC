"""
injeta a cnn substituta (03) como avaliadora de compliancia no dqn de brown
et al. (2022), somente na malha fina 24x24 do refinamento progressivo, e
compara topologia e tempo contra o dqn original e o simp em tres casos.
requer o codigo de brown vendorizado em brown_original/ (top_opt_rl) e
tensorflow. limitacoes declaradas: a cnn nao preve o campo de tensao
(devolvido constante) e os canais de forca entram zerados.
saida: resultados/04_benchmark_brown_surrogate.png
"""

import os
import sys
import time
from importlib.util import spec_from_file_location, module_from_spec

# os checkpoints de Brown foram salvos com o otimizador legado do Keras; em
# TF >= 2.11 o Adam padrao mudou, entao forcamos o legado antes de carregar o modelo
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam

_DIR = os.path.dirname(os.path.abspath(__file__))
_BROWN = os.path.join(_DIR, "brown_original")
sys.path.insert(0, _BROWN)
sys.path.insert(0, _DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unittest.mock import patch
from scipy.ndimage import convolve

# --- fonte casada com o corpo do texto -------------------------------------
LARGURA_FIG_POL = 13.0        # largura do figsize deste script
LARGURA_MANCHA_POL = 6.30     # 16 cm de mancha do USPSC (A4, margens 3 cm / 2 cm)
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

# o solver e a arquitetura vivem em modulos cujo nome comeca com digito;
# carregamos pelos caminhos dos arquivos
_spec = spec_from_file_location("cnn_arch", os.path.join(_DIR, "03_surrogates.py"))
_cnn_mod = module_from_spec(_spec)
_spec.loader.exec_module(_cnn_mod)
SurrogateCNN = _cnn_mod.SurrogateCNN

_spec2 = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod2 = module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
FEMSolver = _mod2.FEMSolver
FE = _mod2.FE

from Top_Opt_RL.DQN.opts import parse_opts
from Top_Opt_RL.DQN.main import EnviromentsRL, TopOpt_Designing
from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import FEASolve

# casos de teste no formato do ambiente de Brown (2022)
TEST_CASES = {
    "1_Cantilever": {"bcs": [1, 553],             "downs": [312],        "vol": 0.35},
    "2_Ponte":      {"bcs": [552, 553, 574, 575], "downs": [11, 12, 13], "vol": 0.40},
    "3_Console":    {"bcs": [1, 289, 553],        "downs": [190],        "vol": 0.50},
}

_CNN_PATH = os.path.join(_DIR, "resultados", "dados", "surrogate_cnn_24x24.pth")
try:
    cnn_model = SurrogateCNN()
    cnn_model.load_state_dict(torch.load(_CNN_PATH, map_location="cpu"))
    cnn_model.eval()
    print(f"CNN carregada: {_CNN_PATH}")
except FileNotFoundError:
    print("erro: cnn nao encontrada. rode 02 e 03 antes.")
    sys.exit(1)


def surrogate_patch_func(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=True):
    """Substitui o solver de Brown pela CNN apenas na malha fina (576 elementos).

    A CNN preve a compliancia escalar; os campos de tensao sao devolvidos
    constantes, pois a CNN nao os preve.
    """
    if len(void_check) == 576:
        # tensor (1, 3, 24, 24): canal 0 densidade, canais 1 e 2 (forca) zerados
        x_arr = np.array(void_check, dtype=np.float32).reshape(24, 24)
        tensor = torch.zeros(1, 3, 24, 24, dtype=torch.float32)
        tensor[0, 0] = torch.from_numpy(x_arr)

        with torch.no_grad():
            log_c = cnn_model(tensor).item()
        pred_energy = float(np.expm1(log_c))  # desfaz o log1p aplicado no treino

        d_vme = np.ones(ex * ey)
        d_vmn = np.ones((ex, ey))
        d_vmnn = np.ones((ex + 1, ey + 1))
        return (0.0, pred_energy, d_vme, d_vmn, d_vmnn)

    # malhas 6x6 e 12x12 continuam usando o solver real
    return FEASolve(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=Stress)


def get_xy_from_brown(idx_1based):
    idx = idx_1based - 1
    return idx % 24, idx // 24


def run_simp(name, config):
    """Baseline SIMP na malha 24x24 para o caso dado."""
    print(f"  [SIMP] {name}...")
    nelx, nely = 24, 24
    volfrac = config["vol"]
    penal = 3.0
    solver = FEMSolver(nelx, nely, penal=penal)

    rmin = 1.5
    r = int(np.ceil(rmin))
    xx, yy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    H = np.maximum(0, rmin - np.sqrt(xx**2 + yy**2))
    H = H / H.sum()

    fixed_dofs = []
    for bc_idx in config["bcs"]:
        bx, by = get_xy_from_brown(bc_idx)
        n = (nely + 1) * bx + by
        fixed_dofs.extend([2 * n, 2 * n + 1])

    f_vec = np.zeros(solver.ndof)
    for load_idx in config["downs"]:
        lx, ly = get_xy_from_brown(load_idx)
        n_load = (nely + 1) * lx + ly
        f_vec[2 * n_load + 1] = -1.0

    fixed_arr = np.unique(fixed_dofs).astype(int)
    x = np.ones((nely, nelx)) * volfrac

    loop, ch = 0, 1.0
    while ch > 0.01 and loop < 200:
        loop += 1
        xp = np.clip(convolve(x, H, mode="reflect"), 0.001, 1.0)
        U = FE(nelx, nely, xp, penal, fixed_dofs=fixed_arr, force_vector=f_vec)
        ue = U[solver.edofMat]
        ce = np.einsum("ij,jk,ik->i", ue, solver.KE, ue)
        ce = ce.reshape((nely, nelx), order="F")
        dc = -penal * (xp ** (penal - 1)) * ce
        dc = convolve(dc, H, mode="reflect")
        l1, l2, move = 0.0, 1e9, 0.2
        xn = x.copy()
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lm = 0.5 * (l2 + l1)
            xn = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0,
                 np.minimum(x + move, x * np.sqrt(np.maximum(0, -dc / lm))))))
            if np.mean(xn) - volfrac > 0:
                l1 = lm
            else:
                l2 = lm
        ch = np.max(np.abs(xn - x))
        x = xn.copy()

    return np.clip(convolve(x, H, mode="reflect"), 0.001, 1.0)


def run_brown(config, use_surrogate):
    """Roda o agente de Brown no caso dado e devolve (topologia, tempo)."""
    label = "SURROGATE CNN" if use_surrogate else "ORIGINAL"
    print(f"  [Brown {label}]...")
    full_conf = {
        "bcs":   [str(b) for b in config["bcs"]],
        "downs": [str(d) for d in config["downs"]],
        "rights": [], "lefts": [], "ups": [],
        "volfraction": config["vol"],
    }
    test_args = ["main.py", "--From_App", "True", "--Load_Checkpoints", "True",
                 "--Progressive_Refinement", "True", "--Time_Trial", "True"]

    with patch.object(sys, "argv", test_args):
        opts = parse_opts()
    envs = EnviromentsRL(opts)

    start = time.perf_counter()
    if use_surrogate:
        with patch("Top_Opt_RL.DQN.main.FEASolve", side_effect=surrogate_patch_func), \
             patch("Top_Opt_RL.DQN.TopOpt_Env_Functions.FEASolve", side_effect=surrogate_patch_func):
            app_plot = TopOpt_Designing(full_conf, opts, envs, [])
    else:
        app_plot = TopOpt_Designing(full_conf, opts, envs, [])

    t = time.perf_counter() - start
    tf.keras.backend.clear_session()

    topo = np.zeros((24, 24))
    if app_plot and "Topology" in app_plot:
        topo = np.array([float(v) for v in app_plot["Topology"][-1]]).reshape(24, 24)
    return topo, t


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _OUT = os.path.join(_HERE, "resultados", "04_benchmark_brown_surrogate.png")
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)

    # titulos e rotulo de speedup encurtados (2026-09-22): com 3 colunas cada
    # painel tem so LARGURA_MANCHA_POL/3 = 2.1 pol. de pagina em fonte real de
    # 12pt (mesmo problema geometrico de 11/12_robustez/12_dfam) - "DQN +
    # Substituto CNN" e "Speedup (malha 24×24): 21.4×" nao cabem e vazavam
    # para fora da figura.
    nc = len(TEST_CASES)
    fig, axs = plt.subplots(nc, 3, figsize=(13, 4 * nc))
    titles = ["SIMP", "DQN (ref.)", "DQN+CNN"]

    for row, (name, config) in enumerate(sorted(TEST_CASES.items())):
        print(f"\n>> Caso: {name}")
        r_simp = run_simp(name, config)
        r_orig, t_orig = run_brown(config, use_surrogate=False)
        r_surr, t_surr = run_brown(config, use_surrogate=True)

        # arredonda os tempos ANTES de calcular o speedup (2026-09-22): antes o
        # rotulo mostrava os segundos arredondados mas o "Speedup" vinha da
        # divisao com precisao total, entao a conta feita a mao a partir do que
        # esta impresso na figura nao fechava (ex.: 81.0/3.9 = 20.8, nao 20.9).
        t_orig_r, t_surr_r = round(t_orig, 1), round(t_surr, 1)
        nome_limpo = f"({row + 1}) {name.split('_', 1)[1]}"  # sem underscore literal
        resultados = [
            (r_simp, nome_limpo, "gray_r"),
            (r_orig, f"{t_orig_r:.1f} s".replace(".", ","), "Blues"),
            (r_surr, f"{t_surr_r:.1f} s".replace(".", ","), "Greens"),
        ]

        for col, (topo, tit, cmap) in enumerate(resultados):
            ax = axs[row, col]
            ax.imshow(topo, cmap=cmap, origin="upper", interpolation="nearest",
                      aspect="equal", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(tit, fontsize=_pt(12))
            for sp in ax.spines.values():
                sp.set_edgecolor("black")
                sp.set_linewidth(1.5)

        if t_orig_r > 0:
            axs[row, 2].set_xlabel(f"Aceleração: {t_orig_r/t_surr_r:.1f}×".replace(".", ","),
                                   fontsize=_pt(12))

    for col, tit in enumerate(titles):
        axs[0, col].set_title(f"{tit}\n{axs[0, col].get_title()}", fontsize=_pt(12))

    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Comparação de Topologias e Tempos de Execução — Malha 24×24\n"
    #              "SIMP vs. DQN (referência) vs. DQN + Substituto CNN | 3 casos estruturais",
    #              fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(_OUT)
    plt.close(fig)
    print(f"salvo: {_OUT}")