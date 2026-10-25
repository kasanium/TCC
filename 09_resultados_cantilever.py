"""
avaliacao do agente ppo do cantilever 20x10 (checkpoint treinado por
08_treinar_ppo.py --problem cantilever --simetria horizontal):
  (a) tabela e curvas de treino (recompensa media movel vs passos);
  (b) comparacao de topologias no dominio 2:1 — simp vs dqn de brown et al.
      (2022, bcs a 1/4 e 3/4 da altura da malha 24x24) vs ppo;
  (c) simp binarizado nativo em 20x10 — mesma malha do agente, para uma
      comparacao de compliancia justa entre metodos (ver nota abaixo).

nota (2026-09-20): ate esta versao, (c) era um upscaling bicubico 20x10 ->
60x30 + polimento simp simetrizado (15 iteracoes) comparado contra simp
puro convergido nativamente em 60x30 (abordagem hibrida, woldseth et al.
2022). o professor apontou que essa comparacao e injusta: o ppo so chega
em 60x30 via upscaling, o simp chega la nativamente. as funcoes antigas
(upsample, polish_simp, simp_puro, posprocessamento) foram desativadas
mais abaixo (nao apagadas) e substituidas por uma binarizacao do simp
continuo ja calculado em 20x10 — mesma malha de treino/avaliacao do
agente, sem upscaling de nenhum dos dois lados.

a compliancia e sempre u^t f (brown et al. 2022), comparavel entre etapas.
o painel do dqn requer brown_original/ (top_opt_rl) + tensorflow; sem eles
o painel e pulado com aviso e o resto roda normalmente.

saidas: resultados/09_curvas_treino_cantilever.png,
        resultados/09_comparacao_brown.png
"""

import os
import sys
import time
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    raise ImportError("sb3-contrib necessario: pip install sb3-contrib")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FEMSolver = _mod.FEMSolver
FEMSolverRL = _mod.FEMSolverRL
FE = _mod.FE
sensitivity_analysis = _mod.sensitivity_analysis
optimality_criteria_update = _mod.optimality_criteria_update
precompute_filter = _mod.precompute_filter

_s05 = spec_from_file_location("rl_env_mask", os.path.join(_DIR, "05_ambiente_rl.py"))
_m05 = module_from_spec(_s05)
_s05.loader.exec_module(_m05)
TopoOptMaskEnv = _m05.TopoOptMaskEnv
rodar_guloso = _m05.rodar_guloso

# os checkpoints referenciam a politica pelo nome de modulo "politica_fcn";
# registrar antes do exec para o maskableppo.load desserializar
_s06 = spec_from_file_location("politica_fcn", os.path.join(_DIR, "06_politica_fcn.py"))
_m06 = module_from_spec(_s06)
sys.modules["politica_fcn"] = _m06
_s06.loader.exec_module(_m06)

# --- fonte casada com o corpo do texto -------------------------------------
# este script gera figuras com larguras de figsize diferentes (9" nas curvas
# de treino, 15" na comparacao com o brown); por isso o fator de escala e
# calculado por figura via _pt_de(W), nao por uma constante global.
LARGURA_MANCHA_POL = 6.30     # 16 cm de mancha do USPSC (A4, margens 3 cm / 2 cm)
def _pt_de(W, p=12): return p * W / LARGURA_MANCHA_POL

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": _pt_de(9), "axes.titlesize": _pt_de(9), "axes.labelsize": _pt_de(9),
    "xtick.labelsize": _pt_de(9), "ytick.labelsize": _pt_de(9),
    "legend.fontsize": _pt_de(9), "figure.titlesize": _pt_de(9),
    "axes.linewidth": 1.5, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


# ---------------------------------------------------------------------------
# parametros
# ---------------------------------------------------------------------------
NELX, NELY = 20, 10
VOL_FRAC = 0.5
PENAL = 3.0
RMIN = 1.5
# precisa bater com o que foi usado no treino do checkpoint
SIMETRIA = "horizontal"

# desativado (2026-09-20): parametros do upscaling 20x10 -> 60x30, ver nota
# no topo do arquivo. mantidos comentados para referencia/reversao.
# NELX_HI, NELY_HI = 60, 30
# RMIN_POLISH = 3.0        # rmin escalado com o refino 3x
# POLISH_ITERS = 15
SIMP_MAX_ITERS = 200

_run_ppo = os.path.join(_DIR, "resultados", "dados", "v21_cantilever_20x10")
_best = os.path.join(_run_ppo, "best_model.zip")
MODEL_PPO = _best if os.path.exists(_best) else os.path.join(
    _run_ppo, "ppo_v21_cantilever_final.zip")
CSV_TREINO = os.path.join(_DIR, "resultados", "logs",
                          "v21_cantilever_20x10", "treino.monitor.csv")
out_dir = os.path.join(_DIR, "resultados")
os.makedirs(out_dir, exist_ok=True)

# indices de brown (1-based, malha 24x24, idx = x*EY + y + 1): engastes a
# 1/4 e 3/4 da altura e carga no centro da borda direita emulam o dominio
# 2:1 no framework quadrado dele
EX, EY = 24, 24
Y_LO = EY * 1 // 4
Y_HI = EY * 3 // 4
BC_1 = 0 * EY + Y_LO + 1
BC_2 = 0 * EY + Y_HI + 1
LOAD = (EX - 1) * EY + (Y_LO + Y_HI) // 2 + 1


# ---------------------------------------------------------------------------
# simp e agente na malha 20x10
# ---------------------------------------------------------------------------

def roda_simp(nx=NELX, ny=NELY, vf=VOL_FRAC):
    """simp classico (sigmund 2001) no cantilever 2:1; compliancia u^t f."""
    solver = FEMSolver(nx, ny, vf, PENAL, RMIN)
    H, Hs = precompute_filter(nx, ny, RMIN)
    x = np.ones(ny * nx, dtype=float) * vf
    for _ in range(SIMP_MAX_ITERS):
        x2d = x.reshape(ny, nx, order="F")
        U = FE(nx, ny, x2d, PENAL, fixed_dofs=solver.fixeddofs, force_vector=solver.F)
        c, dc, _, _ = sensitivity_analysis(nx, ny, x2d, PENAL, U)
        x = optimality_criteria_update(nx, ny, x, vf, dc, H, Hs)
    x2d = x.reshape(ny, nx, order="F")
    U = FE(nx, ny, x2d, PENAL, fixed_dofs=solver.fixeddofs, force_vector=solver.F)
    c = float(np.dot(U, solver.F))
    return x2d, c


def avaliar_agente(zip_path=MODEL_PPO):
    """episodio greedy deterministico; devolve (densidade, c, n_componentes)."""
    env = TopoOptMaskEnv(NELX, NELY, VOL_FRAC, problem_type="cantilever",
                         simetria=SIMETRIA)
    co = {"lr_schedule": lambda _: 3e-4, "clip_range": lambda _: 0.2}
    model = MaskablePPO.load(zip_path, custom_objects=co)
    obs, _ = env.reset()
    done = trunc = False
    while not (done or trunc):
        mask = env.get_action_mask()
        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, _, done, trunc, _ = env.step(act)

    d = env.density.copy()
    solver = FEMSolverRL(NELX, NELY, VOL_FRAC, PENAL, RMIN)
    _, c, _ = solver.solve(d)

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, nc = scipy.ndimage.label(d > 0.5, structure=struct)
    return d, float(c), nc


# ---------------------------------------------------------------------------
# curvas de treino
# ---------------------------------------------------------------------------

def ler_csv_logs(path, smooth=2000):
    """le um monitor.csv do sb3; eixo x em passos (milhoes)."""
    if not os.path.exists(path):
        print(f"  aviso: log nao encontrado — {path}")
        return None, None, None
    import pandas as pd
    df = pd.read_csv(path, comment="#", header=0)
    df.columns = df.columns.str.strip()
    rew = df["r"].values.astype(float)
    lens = df["l"].values.astype(float)
    steps_M = np.cumsum(lens) / 1e6
    if len(rew) > smooth:
        k = np.ones(smooth) / smooth
        rew_sm = np.convolve(rew, k, mode="valid")
        steps_sm = steps_M[smooth - 1:]
    else:
        rew_sm, steps_sm = rew, steps_M
    return steps_M, steps_sm, rew_sm


def plotar_curvas():
    _, steps_sm, rew_sm = ler_csv_logs(CSV_TREINO)
    if steps_sm is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(steps_sm, rew_sm, color="#2980B9", lw=2,
            label="PPO com Mascaramento de Ações")
    ax.set_xlabel("Passos (milhões)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_ylabel("Recompensa total (média móvel)")
    ax.legend(fontsize=_pt_de(9), loc="lower right")
    ax.grid(True, alpha=0.15)
    _spines(ax)
    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Curvas de Treinamento — PPO Cantilever $20\\times10$\n"
    #              "Recompensa total (média móvel)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "09_curvas_treino_cantilever.png")
    plt.savefig(out)
    plt.close()
    print(f"  salvo: {out}")


# ---------------------------------------------------------------------------
# dqn de brown no dominio 2:1
# ---------------------------------------------------------------------------

def _rodar_brown_2to1():
    """roda o dqn de brown (2022) com engastes 2:1; devolve o recorte util."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    _BROWN = os.path.join(_DIR, "brown_original")
    if _BROWN not in sys.path:
        sys.path.insert(0, _BROWN)

    import tensorflow as tf
    # os checkpoints de brown usam o otimizador legado do keras
    tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam
    from unittest.mock import patch
    from Top_Opt_RL.DQN.opts import parse_opts
    from Top_Opt_RL.DQN.main import EnviromentsRL, TopOpt_Designing

    full_conf = {
        "bcs": [str(BC_1), str(BC_2)],
        "downs": [str(LOAD)],
        "rights": [], "lefts": [], "ups": [],
        "volfraction": VOL_FRAC,
    }
    test_args = ["main.py", "--From_App", "True", "--Load_Checkpoints", "True",
                 "--Progressive_Refinement", "True", "--Time_Trial", "True"]

    t0 = time.perf_counter()
    with patch.object(sys, "argv", test_args):
        opts = parse_opts()

    # o codigo de brown usa caminhos relativos; entra no diretorio dele
    _orig = os.getcwd()
    os.chdir(os.path.join(_BROWN, "Top_Opt_RL", "DQN"))
    try:
        envs = EnviromentsRL(opts)
        app_plot = TopOpt_Designing(full_conf, opts, envs, [])
    finally:
        os.chdir(_orig)

    elapsed = time.perf_counter() - t0
    tf.keras.backend.clear_session()

    # topologia flat (576), ordem x-major: elem[x*EY + y] -> [y, x]
    topo_flat = np.array([float(v) for v in app_plot["Topology"][-1]])
    density = topo_flat.reshape(EX, EY).T

    # recorte da faixa util entre os apoios
    d_crop = density[Y_LO:Y_HI + 1, :]
    _, nc = scipy.ndimage.label(d_crop > 0.5,
                                structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    print(f"    brown: {nc} componente(s) | {elapsed:.0f}s")
    return d_crop, nc, elapsed


def comparacao_brown(simp_d, c_simp_bin, ppo_d, c_ppo, nc_ppo):
    # painel (a) exibe o simp binarizado (nao o campo continuo em cinza),
    # coerente com (b) e (c), que ja sao binarios por natureza — 2026-09-21.
    print("\ncomparacao com o dqn de brown (dominio 2:1)...")
    try:
        d_crop, nc_brown, t_brown = _rodar_brown_2to1()
        brown_ok = True
    except Exception as exc:
        print(f"    painel do brown pulado ({type(exc).__name__}: {exc})")
        d_crop = np.ones((Y_HI - Y_LO + 1, EX)) * 0.5
        nc_brown, t_brown = 0, 0.0
        brown_ok = False

    # titulos encurtados e fonte reduzida (2026-09-22): com 3 colunas, cada
    # painel tem so LARGURA_MANCHA_POL/3 = 2.1 pol. de pagina - os titulos
    # com "— 20x10"/"— 24x24" na mesma linha do nome vazavam para o painel
    # vizinho (mesmo problema geometrico de 11/12_robustez/12_dfam).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"wspace": 0.05})

    def _panel(ax, arr, title):
        # aspect="equal" (nao "auto"): com aspect="auto" o imshow estica a
        # imagem para preencher a caixa do subplot (quase quadrada aqui, com
        # 3 colunas), distorcendo o dominio 2:1 do cantilever/PPO (20x10) e
        # invertendo visualmente a proporcao anunciada na legenda (2026-09-22).
        ax.imshow(arr, cmap="gray_r", interpolation="nearest",
                  origin="upper", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=_pt_de(11), pad=6)
        _spines(ax)

    simp_d_bin = _binarizar(simp_d, VOL_FRAC)
    _panel(axes[0], simp_d_bin,
           f"(a) SIMP binarizado\n20$\\times$10 | $C_{{bin}} = {c_simp_bin:.2f}$".replace(".", "{,}"))
    titulo_b = (f"(b) DQN (referência)\n"
                f"24$\\times$24 | {nc_brown} comp. | {t_brown:.0f} s" if brown_ok
                else "(b) DQN (referência)\nindisponível")
    _panel(axes[1], (d_crop > 0.5).astype(float), titulo_b)
    _panel(axes[2], (ppo_d > 0.5).astype(float),
           f"(c) PPO (este trabalho)\n20$\\times$10 | $C = {c_ppo:.2f}$".replace(".", "{,}")
           + f" | {nc_ppo} comp.")

    fig.tight_layout()
    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.subplots_adjust(top=0.82)
    # fig.suptitle("Comparação de Topologias — Cantilever $L/H = 2{:}1$\n"
    #              "SIMP vs. DQN (referência) vs. PPO (este trabalho)", fontsize=13)
    out = os.path.join(out_dir, "09_comparacao_brown.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  salvo: {out}")


# ---------------------------------------------------------------------------
# polimento na malha 60x30
# ---------------------------------------------------------------------------

def _binarizar(x2d, vf=VOL_FRAC):
    """binariza pelo percentil que preserva a fracao de volume alvo."""
    cutoff = np.percentile(x2d.flatten(), (1 - vf) * 100)
    return (x2d >= cutoff).astype(float)


def _compliance_bin(x2d_bin, nx, ny, solver):
    x_fem = np.clip(x2d_bin, 0.001, 1.0)
    U = FE(nx, ny, x_fem, PENAL, fixed_dofs=solver.fixeddofs, force_vector=solver.F)
    return float(np.dot(U, solver.F))


# ---------------------------------------------------------------------------
# DESATIVADO (2026-09-20) — nao apagado, so nao chamado mais em lugar nenhum.
# ver nota no topo do arquivo: comparava simp puro convergido nativamente em
# 60x30 contra o ppo upscaled+polido a partir do 20x10, comparacao apontada
# como injusta (resolucoes diferentes). substituido por simp_binarizado_nativo
# logo abaixo, que binariza o simp continuo ja calculado em 20x10 (mesma
# malha do agente).
# ---------------------------------------------------------------------------
"""
def _simetrizar_flat(arr_flat, ny, nx):
    #media com o espelho vertical; vetor achatado em ordem fortran.
    arr2d = arr_flat.reshape(ny, nx, order="F")
    arr2d = 0.5 * (arr2d + arr2d[::-1, :])
    return arr2d.flatten(order="F")


def upsample(density_rl, nx_hi=NELX_HI, ny_hi=NELY_HI, vf=VOL_FRAC,
             simetria=SIMETRIA):
    #upscaling bicubico, binarizacao e remocao de fragmentos soltos.
    print(f"upscaling bicubico {NELX}x{NELY} -> {nx_hi}x{ny_hi}...")
    upscaled = scipy.ndimage.zoom(density_rl, (ny_hi / NELY, nx_hi / NELX), order=3)
    upscaled = np.clip(upscaled, 0.0, 1.0)
    binary = _binarizar(upscaled, vf)

    struct_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_comp = scipy.ndimage.label(binary, structure=struct_conn)
    if n_comp > 1:
        sizes = scipy.ndimage.sum(binary, labeled, range(1, n_comp + 1))
        binary = (labeled == (np.argmax(sizes) + 1)).astype(float)
    if simetria == "horizontal":
        # o zoom + binarizacao por percentil podem deixar assimetria residual
        # de 1 pixel; copia a metade de cima sobre a de baixo
        metade = binary.shape[0] // 2
        binary[-metade:, :] = binary[:metade, :][::-1, :]
    print(f"  vol = {np.mean(binary):.3f} ({n_comp} comp -> 1)")
    return binary


def polish_simp(binary_hi, nx=NELX_HI, ny=NELY_HI, vf=VOL_FRAC, simetria=SIMETRIA):
    #polimento simp em poucas iteracoes; sob simetria, a sensibilidade e a
    #variavel de projeto sao espelhadas e mediadas a cada iteracao (sem isso
    #o polimento reintroduz assimetria e degrada o resultado).
    print(f"polindo com simp ({POLISH_ITERS} iters) na malha {nx}x{ny}, "
          f"rmin={RMIN_POLISH}...")
    solver = FEMSolver(nx, ny, vf, PENAL, RMIN_POLISH)
    H, Hs = precompute_filter(nx, ny, RMIN_POLISH)

    x = np.clip(binary_hi.flatten(order="F"), 0.001, 1.0)
    for _ in range(POLISH_ITERS):
        x2d = x.reshape(ny, nx, order="F")
        U = FE(nx, ny, x2d, PENAL, fixed_dofs=solver.fixeddofs, force_vector=solver.F)
        _, dc, _, _ = sensitivity_analysis(nx, ny, x2d, PENAL, U)
        if simetria == "horizontal":
            dc = _simetrizar_flat(dc, ny, nx)
        x = optimality_criteria_update(nx, ny, x, vf, dc, H, Hs)
        if simetria == "horizontal":
            x = _simetrizar_flat(x, ny, nx)

    x2d_bin = _binarizar(x.reshape(ny, nx, order="F"), vf)
    if simetria == "horizontal":
        metade = ny // 2
        x2d_bin[-metade:, :] = x2d_bin[:metade, :][::-1, :]
    c = _compliance_bin(x2d_bin, nx, ny, solver)
    print(f"  C_RL_bin = {c:.2f}")
    return x2d_bin, c


def simp_puro(nx=NELX_HI, ny=NELY_HI, vf=VOL_FRAC, simetria=SIMETRIA):
    #simp puro na malha fina ate a convergencia. tambem simetrizado quando
    #a simetria e pedida, para uma comparacao limpa: a diferenca fica so na
    #abordagem (rl vs gradiente), nao em ruido numerico residual.
    print(f"simp puro {nx}x{ny} ate convergencia...")
    solver = FEMSolver(nx, ny, vf, PENAL, RMIN_POLISH)
    H, Hs = precompute_filter(nx, ny, RMIN_POLISH)

    x = vf * np.ones(ny * nx, dtype=float)
    change, loop = 1.0, 0
    while change > 0.01 and loop < SIMP_MAX_ITERS:
        loop += 1
        xold = x.copy()
        x2d = x.reshape(ny, nx, order="F")
        U = FE(nx, ny, x2d, PENAL, fixed_dofs=solver.fixeddofs, force_vector=solver.F)
        _, dc, _, _ = sensitivity_analysis(nx, ny, x2d, PENAL, U)
        if simetria == "horizontal":
            dc = _simetrizar_flat(dc, ny, nx)
        x = optimality_criteria_update(nx, ny, x, vf, dc, H, Hs)
        if simetria == "horizontal":
            x = _simetrizar_flat(x, ny, nx)
        change = np.linalg.norm(x - xold, np.inf)

    x2d_bin = _binarizar(x.reshape(ny, nx, order="F"), vf)
    if simetria == "horizontal":
        metade = ny // 2
        x2d_bin[-metade:, :] = x2d_bin[:metade, :][::-1, :]
    c = _compliance_bin(x2d_bin, nx, ny, solver)
    print(f"  convergiu em {loop} iters | C_SIMP_bin = {c:.2f}")
    return x2d_bin, c, loop
"""


POLISH_ITERS_NATIVO = 15  # iteracoes do polimento em malha nativa (sem upscaling)


def simp_binarizado_nativo(d_simp_continuo, solver_20x10, vf=VOL_FRAC,
                            simetria=SIMETRIA):
    """binariza o simp continuo ja calculado em 20x10 (mesma malha do
    agente), mantem so o maior componente conexo e aplica um polimento
    simp de poucas iteracoes NA PROPRIA malha 20x10 (sem upscaling — nao
    reintroduz a comparacao entre resolucoes distintas apontada como
    injusta). Devolve (topologia_binaria, compliancia). Substitui o
    upscaling+polimento em 60x30 desativado acima — ver nota no topo do
    arquivo. 2026-09-21: adicionado o polimento nativo porque a
    binarizacao pura por percentil, nesta malha grosseira, produzia uma
    forma com vazio encostando no engaste — feia e questionavel, embora
    nao invalida para o solver (as condicoes de contorno sao de
    deslocamento prescrito, nao dependem de haver material ali)."""
    binary = _binarizar(d_simp_continuo, vf)
    struct_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, n_comp = scipy.ndimage.label(binary, structure=struct_conn)
    if n_comp > 1:
        sizes = scipy.ndimage.sum(binary, labeled, range(1, n_comp + 1))
        binary = (labeled == (np.argmax(sizes) + 1)).astype(float)

    H, Hs = precompute_filter(NELX, NELY, RMIN)
    x = np.clip(binary.flatten(order="F"), 0.001, 1.0)
    for _ in range(POLISH_ITERS_NATIVO):
        x2d = x.reshape(NELY, NELX, order="F")
        U = FE(NELX, NELY, x2d, PENAL, fixed_dofs=solver_20x10.fixeddofs,
               force_vector=solver_20x10.F)
        _, dc, _, _ = sensitivity_analysis(NELX, NELY, x2d, PENAL, U)
        x = optimality_criteria_update(NELX, NELY, x, vf, dc, H, Hs)

    x2d_bin = _binarizar(x.reshape(NELY, NELX, order="F"), vf)
    c = _compliance_bin(x2d_bin, NELX, NELY, solver_20x10)
    return x2d_bin, c


def varredura_sementes(c_ref):
    """avalia checkpoints de sementes extras (v21_cantilever_20x10_seed*/)
    e imprime media e desvio do C_greedy — estabilidade do treino."""
    import glob
    pads = sorted(glob.glob(os.path.join(
        _DIR, "resultados", "dados", "v21_cantilever_20x10_seed*",
        "best_model.zip")))
    if not pads:
        return
    print("\nestabilidade entre sementes:")
    print(f"   seed 42 (canonico): C = {c_ref:.4f}")
    cs = [c_ref]
    for p in pads:
        tag = os.path.basename(os.path.dirname(p))
        _, c, _ = avaliar_agente(p)
        print(f"   {tag}: C = {c:.4f}")
        cs.append(c)
    cs = np.array(cs)
    print(f"   media = {cs.mean():.4f} | desvio = {cs.std():.4f} | n = {len(cs)}")


# ---------------------------------------------------------------------------
# DESATIVADO (2026-09-20) — dependia de upsample/polish_simp/simp_puro acima,
# ja desativados. ver nota no topo do arquivo.
# ---------------------------------------------------------------------------
"""
def posprocessamento(density_rl, c_rl, nc_rl):
    binary_hi = upsample(density_rl)

    t0 = time.time()
    rl_bin, c_rl_bin = polish_simp(binary_hi)
    t_rl = time.time() - t0

    # topologia final para as analises de robustez/dfam do 12
    _dados = os.path.join(_DIR, "resultados", "dados")
    os.makedirs(_dados, exist_ok=True)
    np.savez_compressed(os.path.join(_dados, "topologia_ppo_cantilever.npz"),
                        bruto=density_rl, polido=rl_bin)

    t0 = time.time()
    simp_bin, c_simp_hi, n_iters = simp_puro()
    t_simp = time.time() - t0

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    def _show(ax, arr, titulo, cor="black"):
        ax.imshow(1.0 - arr, cmap="gray", interpolation="nearest",
                  aspect="auto", vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=11, color=cor, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)

    _show(axes[0], simp_bin,
          f"(a) SIMP puro — binarizado — $60\\times30$\n"
          f"$C_{{bin}}={c_simp_hi:.2f}$ | iters FEM = {n_iters} | $t={t_simp:.1f}$ s",
          cor="#2980B9")
    _show(axes[1], density_rl > 0.5,
          f"(b) PPO com Mascaramento de Ações — $20\\times10$\n"
          f"$C={c_rl:.2f}$ | {nc_rl} comp.")
    _show(axes[2], binary_hi,
          f"(c) Upscaling + binarização — $60\\times30$\nVF={VOL_FRAC}")
    _show(axes[3], rl_bin,
          f"(d) PPO + polimento SIMP — binarizado — $60\\times30$\n"
          f"$C_{{bin}}={c_rl_bin:.2f}$ | iters FEM = {POLISH_ITERS} | $t={t_rl:.1f}$ s",
          cor="#27AE60")

    fig.suptitle("Pós-Tratamento da Saída do Agente PPO via Polimento SIMP — "
                 "$60\\times30$", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "09_posprocessamento_cantilever.png")
    plt.savefig(out)
    plt.close()
    print(f"  salvo: {out}")

    print("\n=== resumo do polimento ===")
    print(f"RL + polimento binarizado : C = {c_rl_bin:.2f} | t = {t_rl:.1f}s "
          f"({POLISH_ITERS} iters FEM)")
    print(f"SIMP puro binarizado      : C = {c_simp_hi:.2f} | t = {t_simp:.1f}s "
          f"({n_iters} iters FEM)")
    print(f"Delta C                   : {c_rl_bin - c_simp_hi:+.2f}")
    if t_rl > 0:
        print(f"Speedup (tempo de FEM)    : {t_simp / t_rl:.1f}x")
"""


if __name__ == "__main__":
    print("=== avaliacao do cantilever ===\n")
    if not os.path.exists(MODEL_PPO):
        print(f"erro: checkpoint nao encontrado: {MODEL_PPO}")
        print("rode 08_treinar_ppo.py --problem cantilever --simetria horizontal antes.")
        sys.exit(1)

    print("1. simp 20x10 (referencia na malha do agente)...")
    d_simp, c_simp = roda_simp()
    print(f"   C_simp = {c_simp:.4f}")

    # simp binarizado na mesma malha do agente (20x10) — comparacao justa,
    # sem upscaling de nenhum dos dois lados. substitui o antigo passo 5
    # (upscaling 20x10 -> 60x30 + polimento), ver nota no topo do arquivo.
    solver_20x10 = FEMSolver(NELX, NELY, VOL_FRAC, PENAL, RMIN)
    _, c_simp_bin = simp_binarizado_nativo(d_simp, solver_20x10)
    print(f"   C_simp_bin (mesma malha do agente) = {c_simp_bin:.4f}")

    print("\n2. baseline guloso eso (mesmo espaco de acoes e mesma simetria)...")
    # sob simetria o filtro ajuda (78.90 vs 180.69 sem filtro); ver 05
    _, c_eso, _, _ = rodar_guloso(nelx=NELX, nely=NELY, vol_frac=VOL_FRAC,
                                  problem_type="cantilever", raio_filtro=1.5,
                                  seed=0, simetria=SIMETRIA, verbose=False)
    print(f"   C_eso = {c_eso:.4f}")

    print("\n3. agente ppo (greedy deterministico)...")
    d_ppo, c_ppo, nc_ppo = avaliar_agente()
    print(f"   C = {c_ppo:.4f} | ratio = {c_ppo/c_simp:.3f}x | componentes = {nc_ppo}")

    sep = "-" * 66
    print(f"\n{sep}")
    print(f"  cantilever {NELX}x{NELY} | VF = {VOL_FRAC*100:.0f}% | "
          f"simetria = {SIMETRIA}")
    print(sep)
    print(f"  {'Metodo':<32} {'C':>8}  {'C/C_SIMP':>9}  {'Comp.':>6}")
    print(f"  {'SIMP continuo (referencia)':<32} {c_simp:>8.2f}  "
          f"{'1.00x':>9}  {'1':>6}")
    print(f"  {'ESO guloso (simetria, sem treino)':<32} {c_eso:>8.2f}  "
          f"{c_eso/c_simp:>8.2f}x  {'1':>6}")
    print(f"  {'PPO (hard masking + simetria)':<32} {c_ppo:>8.2f}  "
          f"{c_ppo/c_simp:>8.2f}x  {nc_ppo:>6}")
    print(f"  {'SIMP binarizado (mesma malha)':<32} {c_simp_bin:>8.2f}  "
          f"{c_simp_bin/c_simp:>8.2f}x  {'1':>6}")
    print(sep)

    varredura_sementes(c_ppo)

    print("\n4. curvas de treino...")
    plotar_curvas()

    comparacao_brown(d_simp, c_simp_bin, d_ppo, c_ppo, nc_ppo)

    print("\n=== concluido ===")
