"""
demonstracao do padrao em xadrez (checkerboarding, diaz e sigmund 1995) e do
efeito do filtro de sensibilidade de sigmund (2001). roda o mesmo simp/oc do
caso de validacao de kharmanda et al. (2004) (32x20, f=0.4, p=3, 120
iteracoes fixas, sem parada antecipada — mesmo numero de iteracoes nos dois
casos, para a unica diferenca ser o filtro) duas vezes:
  (a) sem filtro: H = I, Hs = 1 (o operador da eq. do filtro vira identidade,
      a sensibilidade filtrada e igual a original);
  (b) com filtro: r_min = 1.2 (mesmo valor do caso de validacao).

indice de xadrez: fracao de elementos internos (com os 4 vizinhos ortogonais
dentro do dominio) cuja classificacao binaria (solido/vazio, limiar 0.5) e
oposta a de todos os 4 vizinhos — a assinatura geometrica do padrao alternado.
medido, nao estimado.

saida: resultados/13_checkerboard.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FE = _mod.FE
sensitivity_analysis = _mod.sensitivity_analysis
optimality_criteria_update = _mod.optimality_criteria_update
precompute_filter = _mod.precompute_filter

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

NELX, NELY = 32, 20
VOLFRAC = 0.4
PENAL = 3.0
RMIN = 1.2
N_ITERS = 120
_OUTDIR = os.path.join(_DIR, "resultados")
os.makedirs(_OUTDIR, exist_ok=True)


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def indice_xadrez(x2d, thresh=0.5):
    """fracao de elementos internos cuja classificacao binaria e oposta a
    dos 4 vizinhos ortogonais (assinatura do padrao em xadrez). x2d tem
    shape (nely, nelx)."""
    solido = x2d > thresh
    ny, nx = solido.shape
    interno = solido[1:-1, 1:-1]
    cima = solido[:-2, 1:-1]
    baixo = solido[2:, 1:-1]
    esq = solido[1:-1, :-2]
    dir_ = solido[1:-1, 2:]
    oposto = (cima != interno) & (baixo != interno) & (esq != interno) & (dir_ != interno)
    n_interno = interno.size
    return float(oposto.sum()) / n_interno if n_interno else 0.0


def roda_simp(rmin, n_iters=N_ITERS):
    """simp/oc de kharmanda (canto inferior direito, engaste na aresta
    esquerda) com rmin dado; rmin<=0 desliga o filtro (H=I, Hs=1)."""
    nelx, nely = NELX, NELY
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    F[ndof - 1] = -1.0

    x = VOLFRAC * np.ones(nely * nelx, dtype=float)
    if rmin > 0:
        H, Hs = precompute_filter(nelx, nely, rmin)
    else:
        n = nelx * nely
        H = np.eye(n)
        Hs = np.ones(n)

    for _ in range(n_iters):
        x2d = x.reshape(nely, nelx, order="F")
        U = FE(nelx, nely, x2d, PENAL, force_vector=F)
        c, dc, _, _ = sensitivity_analysis(nelx, nely, x2d, PENAL, U)
        x = optimality_criteria_update(nelx, nely, x, VOLFRAC, dc, H, Hs)

    x2d = x.reshape(nely, nelx, order="F")
    U = FE(nelx, nely, x2d, PENAL, force_vector=F)
    c = float(np.dot(U, F))
    return x2d, c


if __name__ == "__main__":
    print(f"=== checkerboard: kharmanda {NELX}x{NELY}, f={VOLFRAC}, p={PENAL}, "
          f"{N_ITERS} iteracoes fixas ===\n")

    print("sem filtro (H=I, Hs=1)...")
    x_sem, c_sem = roda_simp(rmin=0.0)
    idx_sem = indice_xadrez(x_sem)
    print(f"  C = {c_sem:.4f} | indice de xadrez = {100*idx_sem:.1f}%")

    print(f"\ncom filtro (r_min={RMIN})...")
    x_com, c_com = roda_simp(rmin=RMIN)
    idx_com = indice_xadrez(x_com)
    print(f"  C = {c_com:.4f} | indice de xadrez = {100*idx_com:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(LARGURA_FIG_POL, 4.2))
    for ax, x2d, c, idx, titulo in [
            (axes[0], x_sem, c_sem, idx_sem, "(a) Sem filtro"),
            (axes[1], x_com, c_com, idx_com,
             f"(b) Com filtro ($r_{{\\min}}={RMIN}$)".replace(".", "{,}"))]:
        ax.imshow(1.0 - x2d, cmap="gray", interpolation="none", aspect="equal",
                  vmin=0, vmax=1)
        c_br = f"{c:.2f}".replace(".", "{,}")
        idx_br = f"{100*idx:.1f}".replace(".", "{,}")
        ax.set_title(f"{titulo}\n$C={c_br}$ | xadrez $={idx_br}\\%$", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)

    fig.tight_layout()
    out = os.path.join(_OUTDIR, "13_checkerboard.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"\nsalvo: {out}")
