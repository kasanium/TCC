"""
baseline simp do suporte automotivo 30x30 e estudo de ablacao.

cada furo de parafuso vira duas regioes passivas: void (raio r_hole, sempre
vazio) e olhal solido (ate r_boss, sempre cheio); o engaste fica so nos nos
da interface do furo, de modo que o olhal deforma e carrega tensao real.
carga horizontal unitaria (tracao) no no central da aresta direita. o criterio de
otimalidade trata os passivos dentro da bisseccao de volume (oc_passive,
andreassen et al. 2011; filtro de sensibilidade de sigmund 2001).

build_bcs() e parametrizavel (posicao/raio dos furos, magnitude/angulo/
ponto da carga) e e reusado por 11 e 12.

saidas: resultados/10_simp_automotivo.png (topologia + von mises),
        resultados/10_ablacao_automotivo.png (6 configuracoes)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FE = _mod.FE
sensitivity_analysis = _mod.sensitivity_analysis
precompute_filter = _mod.precompute_filter

NELX, NELY = 30, 30
VOL_FRAC = 0.3
PENAL = 3.0
RMIN = 1.5
MAX_ITERS = 200
EMIN = 1e-3
RING = 2          # espessura do olhal (em elementos) ao redor de cada furo

_OUTDIR = os.path.join(_DIR, "resultados")
os.makedirs(_OUTDIR, exist_ok=True)

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


def _br(v, nd=2):
    """numero com virgula decimal para mathtext ({,} evita o espaco apos a virgula)."""
    return f"{v:.{nd}f}".replace(".", "{,}")


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def build_bcs(nelx, nely, centers=None, r_hole=None,
              load_angle_deg=0.0, load_scale=1.0, load_shift=0):
    """bcs automotivas parametrizaveis. nominal: geometria de referencia 30x30
    (furos em (5; 7,5) e (5; 22,5), raio 2, olhal de espessura RING), escalada
    em ponto flutuante para a malha atual, de modo que 30x30, 60x60 e 90x90
    representem a mesma peca fisica, simetrica em torno de y = nely/2; carga
    horizontal de tracao no meio da aresta direita (angulo 0 = direcao nominal)."""
    scale = nelx / 30.0
    if centers is None:
        centers = [(5.0 * scale, 7.5 * scale),
                   (5.0 * scale, 22.5 * scale)]
    if r_hole is None:
        r_hole = 2.0 * scale
    r_boss = r_hole + RING * scale

    ndof = 2 * (nelx + 1) * (nely + 1)
    void = np.zeros((nely, nelx), dtype=bool)
    solid = np.zeros((nely, nelx), dtype=bool)
    for ely in range(nely):
        for elx in range(nelx):
            cx, cy = elx + 0.5, ely + 0.5
            for (hx, hy) in centers:
                d2 = (cx - hx) ** 2 + (cy - hy) ** 2
                if d2 <= r_hole ** 2:
                    void[ely, elx] = True
                elif d2 <= r_boss ** 2:
                    solid[ely, elx] = True
    solid &= ~void   # o furo tem prioridade na sobreposicao

    # engaste apenas nos nos dos elementos void (interface do furo): o olhal
    # deforma e carrega tensao; fixar o disco inteiro mataria essa tensao
    fixed_nodes = set()
    for ely, elx in np.argwhere(void):
        for (cx, cy) in ((elx, ely), (elx + 1, ely),
                         (elx, ely + 1), (elx + 1, ely + 1)):
            fixed_nodes.add((nely + 1) * cx + cy)
    fixed_nodes = np.array(sorted(fixed_nodes), dtype=int)
    fixeddofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    F = np.zeros(ndof)
    node_row = min(max(nely // 2 + load_shift, 0), nely)
    load_node = (nely + 1) * nelx + node_row
    th = np.deg2rad(load_angle_deg)
    F[2 * load_node] = load_scale * np.sin(th)
    F[2 * load_node + 1] = -load_scale * np.cos(th)

    return dict(fixeddofs=fixeddofs, freedofs=freedofs, F=F,
                void=void, solid=solid, centers=centers,
                r_hole=r_hole, r_boss=r_boss)


def build_bcs_sem_furos(nelx, nely):
    """caso de controle da ablacao: dominio cheio, engaste na aresta
    esquerda inteira e carga vertical no meio da direita."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    fixeddofs = np.arange(2 * (nely + 1))
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    F = np.zeros(ndof)
    load_node = (nely + 1) * nelx + nely // 2
    F[2 * load_node + 1] = -1.0
    return dict(fixeddofs=fixeddofs, freedofs=freedofs, F=F,
                void=np.zeros((nely, nelx), dtype=bool),
                solid=np.zeros((nely, nelx), dtype=bool),
                centers=[], r_hole=0, r_boss=0)


def oc_passive(x, volfrac, dc, H, Hs, solid_flat, void_flat, move=0.2):
    """criterio de otimalidade com passivos (olhal cheio, furo vazio)
    impostos dentro da bisseccao do volume, para a fracao continuar certa."""
    dcf = (H @ (x * dc)) / (Hs * x)
    l1, l2 = 0.0, 1e5
    xnew = x.copy()
    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = x * np.sqrt(np.maximum(0.0, -dcf) / lmid)
        xnew = np.maximum(EMIN, np.maximum(x - move,
               np.minimum(1.0, np.minimum(x + move, step))))
        xnew[solid_flat] = 1.0
        xnew[void_flat] = EMIN
        if xnew.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def rodar_simp(bc, nelx=NELX, nely=NELY, volfrac=VOL_FRAC, rmin=RMIN,
               max_iters=MAX_ITERS, verbose=False):
    """simp ate a convergencia sob as bcs dadas; retorna (x2d, c, loop)."""
    void_flat = bc["void"].flatten(order="F")
    solid_flat = bc["solid"].flatten(order="F")
    H, Hs = precompute_filter(nelx, nely, rmin)
    x = np.full(nelx * nely, volfrac, dtype=float)
    x[solid_flat] = 1.0
    x[void_flat] = EMIN
    change, loop, c = 1.0, 0, 0.0
    while change > 0.01 and loop < max_iters:
        loop += 1
        xold = x.copy()
        x2d = x.reshape(nely, nelx, order="F")
        U = FE(nelx, nely, x2d, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
        c, dc, _, _ = sensitivity_analysis(nelx, nely, x2d, PENAL, U)
        x = oc_passive(x, volfrac, dc, H, Hs, solid_flat, void_flat)
        change = np.linalg.norm(x - xold, np.inf)
        if verbose and (loop % 10 == 0 or change <= 0.01):
            print(f"  iter {loop:3d} | C={c:.3f} | vol={x.mean():.3f} | mudanca={change:.4f}")
    return x.reshape(nely, nelx, order="F"), c, loop


def _demarcar_furos(ax, bc):
    for (hx, hy) in bc["centers"]:
        ax.add_patch(Circle((hx - 0.5, hy - 0.5), bc["r_hole"],
                            fill=False, lw=1.4, ec="#C0392B", ls="-"))
        ax.add_patch(Circle((hx - 0.5, hy - 0.5), bc["r_boss"],
                            fill=False, lw=1.4, ec="#27AE60", ls="--"))


def baseline():
    """caso nominal (v=30%, rmin=1.5): topologia + von mises."""
    print(f"simp automotivo {NELX}x{NELY} | VF={VOL_FRAC} | rmin={RMIN} | ring={RING}")
    bc = build_bcs(NELX, NELY)
    void_flat = bc["void"].flatten(order="F")
    solid_flat = bc["solid"].flatten(order="F")

    n = NELX * NELY
    vmin = solid_flat.sum() / n
    vmax = 1.0 - void_flat.sum() / n
    print(f"  furo={void_flat.sum()} elem | olhal={solid_flat.sum()} elem | "
          f"VF viavel em [{vmin:.3f}, {vmax:.3f}]")
    if not (vmin <= VOL_FRAC <= vmax):
        raise ValueError(f"VF={VOL_FRAC} fora da faixa viavel [{vmin:.3f},{vmax:.3f}]")

    x2d, c, loop = rodar_simp(bc, verbose=True)
    print(f"\ncompliance final: {c:.4f} | iters: {loop} | vol_final={x2d.mean():.3f}")

    U = FE(NELX, NELY, x2d, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
    _, _, _, vm_map = sensitivity_analysis(NELX, NELY, x2d, PENAL, U)
    vm_map = vm_map.copy()
    vm_map[bc["void"]] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(x2d, cmap="gray_r", interpolation="none", aspect="equal", vmin=0, vmax=1)
    axes[0].set_title(f"(a) Topologia | $C = {_br(c, 4)}$", fontweight="bold")
    _demarcar_furos(axes[0], bc)

    vm_norm = vm_map / (vm_map.max() + 1e-12)
    im2 = axes[1].imshow(vm_norm, cmap="turbo", interpolation="none",
                         aspect="equal", vmin=0, vmax=1)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.set_label("Von Mises normalizado", fontsize=_pt(12))
    cb2.ax.tick_params(labelsize=_pt(12))
    cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.1f}".replace(".", ",")))
    axes[1].set_title("(b) Von Mises (aprox.)", fontweight="bold")
    _demarcar_furos(axes[1], bc)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)

    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle(rf"SIMP — Suporte {NELX}$\times${NELY}"
    #              "\nTopologia e campo de tensão de von Mises", fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(_OUTDIR, "10_simp_automotivo.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"salvo: {out}")
    return x2d, c, bc


def ablacao():
    """seis configuracoes: rmin, fracao de volume e controle sem furos.
    ordem sistematica (2026-09-22, resposta a critica de banca sobre series
    entrelacadas): linha 1 = volume crescente (rmin fixo), linha 2 = rmin
    crescente (volume fixo) + controle."""
    configs = [
        {"label": "$r_{\\min}=1{,}5$\n$V=20\\%$",            "vol": 0.2, "rmin": 1.5, "furos": True},
        {"label": "$r_{\\min}=1{,}5$\n$V=30\\%$ (referência)", "vol": 0.3, "rmin": 1.5, "furos": True},
        {"label": "$r_{\\min}=1{,}5$\n$V=40\\%$",            "vol": 0.4, "rmin": 1.5, "furos": True},
        {"label": "$r_{\\min}=3{,}0$\n$V=30\\%$",            "vol": 0.3, "rmin": 3.0, "furos": True},
        {"label": "$r_{\\min}=5{,}0$\n$V=30\\%$",            "vol": 0.3, "rmin": 5.0, "furos": True},
        {"label": "Sem furos\n$V=30\\%$ (controle)",        "vol": 0.3, "rmin": 1.5, "furos": False},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.05})
    axes = axes.flatten()

    for i, cfg in enumerate(configs):
        label_clean = cfg["label"].replace("\n", ", ")
        print(f"[{i+1}/{len(configs)}] {label_clean} ...", end=" ", flush=True)
        bc = build_bcs(NELX, NELY) if cfg["furos"] else build_bcs_sem_furos(NELX, NELY)
        x2d, C, loop = rodar_simp(bc, volfrac=cfg["vol"], rmin=cfg["rmin"])
        print(f"C={C:.2f} | {loop} iters")

        ax = axes[i]
        ax.imshow(x2d, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"({i+1}) {cfg['label']}\n$C={_br(C)}$ | {loop} iterações",
                     fontsize=_pt(12), pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)

    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle(f"Estudo de Ablação — Suporte {NELX}$\\times${NELY}\n"
    #              "Efeito do filtro de sensibilidade ($r_{\\min}$) e da fração de volume",
    #              fontsize=13, y=1.01)
    out = os.path.join(_OUTDIR, "10_ablacao_automotivo.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"salvo: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=str, default="tudo",
                    choices=["baseline", "ablacao", "tudo"])
    args = ap.parse_args()
    if args.so in ("baseline", "tudo"):
        baseline()
    if args.so in ("ablacao", "tudo"):
        ablacao()
