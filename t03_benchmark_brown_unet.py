"""
injeta a u-net de campo (t02) no lugar do mef de brown 22 na
malha fina 24x24 do refinamento progressivo e compara com o dqn original e o
simp nos tres casos do 04. diferente do 04, o substituto devolve o campo de
tensao no formato que o agente usa como observacao, e nao um campo constante.
alem do tempo, o script conta as chamadas ao mef e ao substituto na malha
24x24, mede quantos elementos o agente removeu na etapa fina e reavalia a
topologia final com o mef real (a energia do substituto nao entra nessa conta).
metas de volume: cantilever 0,35, ponte e console 0,45 (escolhidas so com o mef,
antes da u-net; ver metas_escolha.csv). --modelo e --sufixo escolhem o modelo
(_A: treinado tambem com episodios do benchmark; _B: --holdout do t02).
requer brown_original/ (ver README.md), tensorflow e o modelo do t02.
saidas: resultados/t03_benchmark_brown_unet.png,
resultados/t03_benchmark_brown_unet.csv
"""

import os
import sys
import csv
import time
from importlib.util import spec_from_file_location, module_from_spec

# os checkpoints de brown foram salvos com o otimizador legado do keras; em
# tf >= 2.11 o adam padrao mudou, entao forcamos o legado antes de carregar o modelo
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
from scipy.stats import spearmanr

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

# a u-net e o formato de saida vivem em t02_unet_tensao.py; o solver do simp
# em 01_fem_simp.py. carregamos pelos caminhos dos arquivos
_spec = spec_from_file_location("unet_tensao", os.path.join(_DIR, "t02_unet_tensao.py"))
_unet = module_from_spec(_spec)
_spec.loader.exec_module(_unet)

_spec2 = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod2 = module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
FEMSolver = _mod2.FEMSolver
FE = _mod2.FE

from Top_Opt_RL.DQN.opts import parse_opts
from Top_Opt_RL.DQN.main import EnviromentsRL, TopOpt_Designing
from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import FEASolve
from Top_Opt_RL.DQN.Matrix_Transforms import Mesh_Transform

# casos de teste no formato do ambiente de brown 22 — mesmos do 04
TEST_CASES = {
    "1_Cantilever": {"bcs": [1, 553],             "downs": [312],        "vol": 0.35},
    "2_Ponte":      {"bcs": [552, 553, 574, 575], "downs": [11, 12, 13], "vol": 0.40},
    "3_Console":    {"bcs": [1, 289, 553],        "downs": [190],        "vol": 0.50},
}

# uma thread: previsao de uma amostra por vez, como no laco do agente; mais
# threads so adicionam sincronizacao numa rede deste tamanho
N_THREADS_TORCH = 1
torch.set_num_threads(N_THREADS_TORCH)

# argumentos lidos ja na importacao (o diag_parada.py importa este modulo):
# --modelo escolhe o .pth; --sufixo separa csv e figura de cada modelo; as
# metas de volume da ponte e do console sao as escolhidas so com o mef
# (metas_escolha.py), antes de rodar a u-net
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--modelo", default=os.path.join(_DIR, "resultados", "dados",
                                                  "unet_tensao_brown_24x24.pth"))
_ap.add_argument("--sufixo", default="_A")
_ap.add_argument("--vol_ponte", type=float, default=0.45)
_ap.add_argument("--vol_console", type=float, default=0.45)
ARGS, _ = _ap.parse_known_args()
if ARGS.vol_ponte is not None:
    TEST_CASES["2_Ponte"]["vol"] = ARGS.vol_ponte
if ARGS.vol_console is not None:
    TEST_CASES["3_Console"]["vol"] = ARGS.vol_console

_UNET_PATH = ARGS.modelo
try:
    unet, unet_norm = _unet.carregar_unet(_UNET_PATH)
    print(f"U-Net carregada: {_UNET_PATH}")
except FileNotFoundError:
    print("erro: u-net nao encontrada. rode t01 e t02 antes.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# contadores e substituicao do solver
# ---------------------------------------------------------------------------

CONT = {"mef_576": 0, "sub_576": 0, "mef_outros": 0, "mef_ref": 0,
        "t_mef_576": 0.0, "t_sub_576": 0.0, "t_mef_outros": 0.0, "t_mef_ref": 0.0}


def zera_contadores():
    for k in CONT:
        CONT[k] = 0 if k.startswith(("mef", "sub")) else 0.0


def fea_contado(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=True):
    """FEASolve real, com contagem e tempo por malha (usado na referencia)."""
    t0 = time.perf_counter()
    res = FEASolve(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=Stress)
    dt = time.perf_counter() - t0
    if ex * ey == 576:
        CONT["mef_576"] += 1
        CONT["t_mef_576"] += dt
    else:
        CONT["mef_outros"] += 1
        CONT["t_mef_outros"] += dt
    return res


def substituto_unet(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=True):
    """substitui o solver de brown pela u-net apenas na malha fina (576 elementos).

    a u-net devolve energia e campo de tensao; saida_formato_brown aplica a
    mesma normalizacao de FEASolve, entao o agente recebe a observacao no
    formato que espera (vazios em 0, elemento menos solicitado em 1).
    """
    if ex * ey == 576:
        t0 = time.perf_counter()
        solido = (np.asarray(void_check, dtype=np.float64).ravel() > 0.5).astype(np.float32)
        if solido.all():
            # bloco cheio: brown usa esta energia como referencia (Max_SE_Tot) no
            # criterio de parada e na recompensa de todo o episodio. e uma chamada
            # por episodio, fora da distribuicao de treino (vf = 1), e um erro nela
            # encerra o episodio na primeira jogada; fica com o mef real
            res = FEASolve(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=Stress)
            CONT["mef_ref"] += 1
            CONT["t_mef_ref"] += time.perf_counter() - t0
            return res
        mapas = _unet.mapas_nos(lc_nodes, load_dirs, bc_nodes, ex, ey)
        se, vm_el = _unet.prever_brown(unet, unet_norm, solido, mapas)
        res = _unet.saida_formato_brown(se, vm_el, ex, ey)
        CONT["sub_576"] += 1
        CONT["t_sub_576"] += time.perf_counter() - t0
        return res

    # malhas 6x6 e 12x12 continuam usando o solver real
    return fea_contado(void_check, lx, ly, ex, ey, lc_nodes, load_dirs, bc_nodes, Stress=Stress)


def get_xy_from_brown(idx_1based):
    idx = idx_1based - 1
    return idx % 24, idx // 24


def bloco2x2(t):
    """fracao dos blocos 2x2 uniformes: 1,00 = topologia 12x12 so ampliada."""
    b = (np.asarray(t) > 0.5).reshape(12, 2, 12, 2).transpose(0, 2, 1, 3).reshape(144, 4)
    return float(np.mean([x.all() or (~x).all() for x in b]))


def run_simp(name, config):
    """baseline simp na malha 24x24 para o caso dado."""
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
    """roda o agente de brown no caso dado e devolve (topologia, tempo, metricas)."""
    label = "SURROGATE U-Net" if use_surrogate else "ORIGINAL"
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

    # os dois modos passam pelo mesmo patch, para o custo do mock ser igual
    func = substituto_unet if use_surrogate else fea_contado
    zera_contadores()
    start = time.perf_counter()
    with patch("Top_Opt_RL.DQN.main.FEASolve", side_effect=func), \
         patch("Top_Opt_RL.DQN.TopOpt_Env_Functions.FEASolve", side_effect=func):
        TopOpt_Designing(full_conf, opts, envs, [])
    t = time.perf_counter() - start
    cont = dict(CONT)
    tf.keras.backend.clear_session()

    final = np.asarray(envs.env.VoidCheck, dtype=float).ravel()
    inicial = np.asarray(Mesh_Transform(opts.PR2_EX, opts.PR2_EY, opts.Main_EX, opts.Main_EY,
                                        envs.env_primer2.VoidCheck), dtype=float).ravel()
    # energia da topologia final pelo mef real, fora do patch, nos dois modos
    res_real = FEASolve(list(final), opts.Lx, opts.Ly, opts.Main_EX, opts.Main_EY,
                        envs.env.LC_Nodes, envs.env.Load_Directions,
                        envs.env.BC_Nodes, Stress=True)
    se_real = float(np.max(res_real[1]))

    # o que a u-net diria dessa mesma topologia final (criterio: acerta no que importa)
    sol = (final > 0.5).astype(np.float32)
    mapas = _unet.mapas_nos(envs.env.LC_Nodes, envs.env.Load_Directions, envs.env.BC_Nodes)
    se_unet, vm_u = _unet.prever_brown(unet, unet_norm, sol, mapas)
    obs_u = _unet.reconstruir_vm_nn(vm_u).ravel()
    obs_r = np.asarray(res_real[3]).ravel()
    rho = spearmanr(obs_u[sol > 0], obs_r[sol > 0]).correlation

    met = {
        "tempo_s": t,
        "vf_final": float(final.mean()),
        "blocos2x2": bloco2x2(final),
        "remocoes_24": int(np.sum((inicial > 0.5) & (final < 0.5))),
        "se_real": se_real,
        "se_unet_final": float(se_unet),
        "erro_se_final_pct": float(100 * (se_unet - se_real) / se_real),
        "spearman_obs_final": float(rho),
        **cont,
    }
    return final.reshape(24, 24), t, met


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _OUT = os.path.join(_HERE, "resultados", f"t03_benchmark_brown_unet{ARGS.sufixo}.png")
    _CSV = os.path.join(_HERE, "resultados", f"t03_benchmark_brown_unet{ARGS.sufixo}.csv")
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    print(f"modelo: {_UNET_PATH} | metas: " +
          ", ".join(f"{k} {v['vol']:.2f}" for k, v in sorted(TEST_CASES.items())))

    # aquecimento fora da medicao: a primeira execucao carrega o tensorflow e
    # compila o grafo, o que penalizaria quem rodasse primeiro (no 04, o original)
    print(">> aquecimento (nao medido)")
    run_brown(TEST_CASES["3_Console"], use_surrogate=False)
    run_brown(TEST_CASES["3_Console"], use_surrogate=True)

    # titulos e rotulo de speedup curtos: com 3 colunas cada painel tem so
    # largura_mancha_pol/3 = 2.1 pol. de pagina em fonte real de 12 pt
    nc = len(TEST_CASES)
    fig, axs = plt.subplots(nc, 3, figsize=(13, 4 * nc))
    titles = ["SIMP", "DQN (ref.)", "DQN+U-Net"]
    linhas = []

    for row, (name, config) in enumerate(sorted(TEST_CASES.items())):
        print(f"\n>> Caso: {name}")
        r_simp = run_simp(name, config)
        r_orig, t_orig, m_orig = run_brown(config, use_surrogate=False)
        r_surr, t_surr, m_surr = run_brown(config, use_surrogate=True)

        for metodo, m in [("DQN (ref.)", m_orig), ("DQN+U-Net", m_surr)]:
            print(f"  {metodo:<11} | VF alvo {config['vol']:.2f} | VF final {m['vf_final']:.3f} | "
                  f"blocos2x2 {m['blocos2x2']:.2f} | remocoes 24x24 {m['remocoes_24']:3d} | "
                  f"MEF 24x24 {m['mef_576']:3d} ({m['t_mef_576']:.1f}s) | "
                  f"U-Net 24x24 {m['sub_576']:3d} ({m['t_sub_576']:.2f}s) | "
                  f"MEF 6x6/12x12 {m['mef_outros']:3d} ({m['t_mef_outros']:.1f}s) | "
                  f"energia (MEF real) {m['se_real']:.4g} | tempo {m['tempo_s']:.1f}s")
            linhas.append({"caso": name, "metodo": metodo, "vf_alvo": config["vol"],
                           **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}})

        # criterios combinados com a vicky (ver LEIA-ME_tentativa.txt, secao 8)
        razao_se = m_surr["se_real"] / m_orig["se_real"]
        rem_u, rem_r = m_surr["remocoes_24"], m_orig["remocoes_24"]
        c_mef_ms = 1e3 * m_orig["t_mef_576"] / max(m_orig["mef_576"], 1)
        c_sub_ms = 1e3 * m_surr["t_sub_576"] / max(m_surr["sub_576"], 1)
        t_passo_r = m_orig["t_mef_576"] / max(rem_r, 1)
        t_passo_u = m_surr["t_sub_576"] / max(rem_u, 1)
        print(f"  criterios ({name}):")
        print(f"    1 etapa 24x24 aconteceu : U-Net chamada {m_surr['sub_576']}x, "
              f"MEF 24x24 no modo U-Net = {m_surr['mef_576']} (tem de ser 0) + "
              f"{m_surr['mef_ref']} do bloco cheio de referencia ({m_surr['t_mef_ref']:.2f}s) | "
              f"remocoes {rem_u} (ref. {rem_r})")
        print(f"    2 VF na meta            : U-Net {m_surr['vf_final']:.3f} | ref. "
              f"{m_orig['vf_final']:.3f} | meta {config['vol']:.2f}")
        print(f"    3 nao e a 12x12 ampliada: blocos2x2 U-Net {m_surr['blocos2x2']:.2f} | "
              f"ref. {m_orig['blocos2x2']:.2f} (1,00 = so ampliada)")
        print(f"    4 topologia boa         : energia (MEF real) U-Net/ref = {razao_se:.3f} "
              f"(alguns % acima de 1,00 = ok; compare so com VF igual)")
        print(f"    5 ganho com trabalho igual: custo por chamada 24x24 MEF {c_mef_ms:.0f} ms x "
              f"U-Net {c_sub_ms:.1f} ms ({c_mef_ms / max(c_sub_ms, 1e-9):.0f}x) | "
              f"etapa 24x24 por remocao: {t_passo_r:.2f} s x {t_passo_u:.3f} s | "
              f"tempo total {m_orig['tempo_s']:.1f} s x {m_surr['tempo_s']:.1f} s "
              f"(6x6/12x12 seguem no MEF nos dois)")
        print(f"    6 acerta no que importa : energia final U-Net x MEF = "
              f"{m_surr['erro_se_final_pct']:+.1f}% | Spearman da observacao na "
              f"topologia final = {m_surr['spearman_obs_final']:.3f}")

        # arredonda os tempos antes de calcular o speedup, para o rotulo bater
        # com a conta feita a mao a partir dos tempos impressos na figura
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

        if t_orig_r > 0 and t_surr_r > 0:
            axs[row, 2].set_xlabel(
                f"Aceleração: {t_orig_r/t_surr_r:.1f}×\n".replace(".", ",")
                + f"$C/C_{{ref}}$ = {razao_se:.2f}".replace(".", ","),
                fontsize=_pt(12))

    for col, tit in enumerate(titles):
        axs[0, col].set_title(f"{tit}\n{axs[0, col].get_title()}", fontsize=_pt(12))

    fig.tight_layout()
    fig.savefig(_OUT)
    plt.close(fig)
    print(f"salvo: {_OUT}")

    with open(_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"salvo: {_CSV}")
