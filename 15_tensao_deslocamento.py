"""
verificacao pos-otimizacao de tensao e deslocamento no suporte automotivo
30x30. as quatro topologias comparadas no 11 (simp binarizado, heuristica
eso, ppo bruto e ppo polido) sao reavaliadas no mesmo mef (bcs do 10, carga
unitaria, e = 1, nu = 0,3): von mises no centro de cada elemento solido
(mesma conta de 01_fem_simp.py) e deslocamentos nodais. como o overhang do
12, e medicao feita depois da otimizacao: nenhum dos metodos tem restricao
de tensao ou de deslocamento.

metricas (adimensionais; o texto as relata relativas ao simp binarizado):
  u_carga    deslocamento do no de aplicacao da carga
  u_max      maior deslocamento entre os nos dos elementos solidos
  s_max      maior von mises entre os elementos solidos
  s_max_fora idem, sem os elementos que tocam o no de carga ou um no
             engastado: carga pontual e engaste nodal concentram tensao que
             cresce com o refinamento da malha
  s_pn       media-p (p = 8) do von mises dos elementos solidos, medida
             global que aproxima o maximo (deaton 14; tang 24)

requer resultados/dados/topologia_ppo_automotivo.npz (gerado pelo 11).
saidas: resultados/15_tensao_deslocamento.{png,csv}
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FE = _mod.FE
sensitivity_analysis = _mod.sensitivity_analysis

# o 11 ja junta as bcs e o simp do 10, a heuristica eso do 05 e a
# binarizacao a vf exato; reusar garante as mesmas quatro topologias
_s11 = spec_from_file_location("resultados_auto", os.path.join(_DIR, "11_resultados_automotivo.py"))
_m11 = module_from_spec(_s11)
_s11.loader.exec_module(_m11)
build_bcs = _m11.build_bcs
rodar_simp = _m11.rodar_simp
rodar_guloso = _m11.rodar_guloso
binarizar_vf = _m11.binarizar_vf
_add_holes = _m11._add_holes
FILTRO_ESO = _m11.FILTRO_ESO

NELX, NELY, VOL_FRAC = 30, 30, 0.3
PENAL, EMIN = 3.0, 1e-3
P_NORMA = 8

_NPZ = os.path.join(_DIR, "resultados", "dados", "topologia_ppo_automotivo.npz")
_OUTDIR = os.path.join(_DIR, "resultados")
os.makedirs(_OUTDIR, exist_ok=True)

# --- fonte casada com o corpo do texto -------------------------------------
LARGURA_FIG_POL = 12.0        # largura do figsize deste script
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


def _br(v, nd=2):
    """numero com virgula decimal para mathtext ({,} evita o espaco apos a virgula)."""
    return f"{v:.{nd}f}".replace(".", "{,}")


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def _nos_para_grade(nos, nelx=NELX, nely=NELY):
    """mascara (nely+1, nelx+1) dos nos dados; no n = (nely+1)*elx + ely."""
    g = np.zeros((nely + 1, nelx + 1), dtype=bool)
    g[nos % (nely + 1), nos // (nely + 1)] = True
    return g


def _elementos_que_tocam(g):
    """elementos (nely, nelx) com algum dos 4 nos marcados na grade g."""
    return g[:-1, :-1] | g[1:, :-1] | g[:-1, 1:] | g[1:, 1:]


def _nos_dos_elementos(s):
    """grade de nos (nely+1, nelx+1) pertencentes aos elementos marcados em s."""
    g = np.zeros((s.shape[0] + 1, s.shape[1] + 1), dtype=bool)
    g[:-1, :-1] |= s
    g[1:, :-1] |= s
    g[:-1, 1:] |= s
    g[1:, 1:] |= s
    return g


def avaliar(x2d, bc):
    """um solve de mef na topologia binaria; devolve as metricas e o campo
    de von mises dos solidos (nan nos vazios, para a figura)."""
    solido = x2d > 0.5
    xb = np.where(solido, 1.0, EMIN)
    U = FE(NELX, NELY, xb, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
    c, _, _, vm = sensitivity_analysis(NELX, NELY, xb, PENAL, U)

    # deslocamento nodal, na mesma grade dos nos
    u_no = np.hypot(U[0::2], U[1::2]).reshape((NELY + 1, NELX + 1), order="F")
    no_carga = int(np.flatnonzero(bc["F"])[0] // 2)
    u_carga = float(u_no[no_carga % (NELY + 1), no_carga // (NELY + 1)])
    u_max = float(u_no[_nos_dos_elementos(solido)].max())

    # zonas de concentracao: elementos que tocam o no de carga ou um no engastado
    fixos = bc["fixeddofs"][bc["fixeddofs"] % 2 == 0] // 2
    toca_carga = _elementos_que_tocam(_nos_para_grade(np.array([no_carga])))
    toca_apoio = _elementos_que_tocam(_nos_para_grade(fixos))

    vm_s = vm[solido]
    s_max = float(vm_s.max())
    s_max_fora = float(vm[solido & ~toca_carga & ~toca_apoio].max())
    s_pn = float(np.mean(vm_s ** P_NORMA) ** (1.0 / P_NORMA))

    ely, elx = np.unravel_index(np.argmax(np.where(solido, vm, -1.0)), vm.shape)
    local = ("carga" if toca_carga[ely, elx] else
             "apoio" if toca_apoio[ely, elx] else "interior")

    vm_fig = np.where(solido, vm, np.nan)
    return dict(C=c, u_carga=u_carga, u_max=u_max, s_max=s_max,
                s_max_fora=s_max_fora, s_pn=s_pn, local=local), vm_fig


def topologias(bc, seed=0):
    """as quatro topologias do 11, todas a vf = 30%."""
    x_simp, _, _ = rodar_simp(bc)
    xb_simp = binarizar_vf(x_simp, bc)
    x_eso, _, _, _ = rodar_guloso(
        nelx=NELX, nely=NELY, vol_frac=VOL_FRAC, problem_type="automotive",
        raio_filtro=FILTRO_ESO, seed=seed, verbose=False)
    ppo = np.load(_NPZ)
    return [
        ("SIMP binarizado", xb_simp),
        ("Heurística ESO", x_eso),
        ("PPO bruto", ppo["bruto"]),
        ("PPO polido", ppo["polido"]),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 74)
    print(" VERIFICACAO DE TENSAO E DESLOCAMENTO — SUPORTE AUTOMOTIVO 30x30")
    print("=" * 74)
    if not os.path.exists(_NPZ):
        print(f"\nerro: {_NPZ} nao encontrado")
        print("rode 11_resultados_automotivo.py antes.")
        sys.exit(1)

    bc = build_bcs(NELX, NELY)
    res = []
    for nome, x2d in topologias(bc, args.seed):
        m, vm = avaliar(x2d, bc)
        res.append((nome, x2d, m, vm))

    ref = res[0][2]
    chaves = ["C", "u_carga", "u_max", "s_max", "s_max_fora", "s_pn"]
    sep = "-" * 86
    print(f"\nvalores absolutos (e = 1, carga unitaria, p = {P_NORMA}):")
    print(sep)
    print(f"  {'topologia':<18}" + "".join(f"{k:>11}" for k in chaves) + f"{'local s_max':>13}")
    print(sep)
    for nome, _, m, _ in res:
        print(f"  {nome:<18}" + "".join(f"{m[k]:>11.4f}" for k in chaves) + f"{m['local']:>13}")
    print(sep)
    print("\nrelativos ao SIMP binarizado (%):")
    print(sep)
    for nome, _, m, _ in res:
        print(f"  {nome:<18}" + "".join(f"{100 * (m[k] / ref[k] - 1):>+10.1f}%" for k in chaves))
    print(sep)

    linhas = [("topologia",) + tuple(chaves) + ("local_s_max",)]
    for nome, _, m, _ in res:
        linhas.append((nome,) + tuple(f"{m[k]:.4f}" for k in chaves) + (m["local"],))
    csv_path = os.path.join(_OUTDIR, "15_tensao_deslocamento.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"\ntabela: {csv_path}")

    # escala de cor comum, limitada ao maior pico fora das zonas de
    # concentracao: o pico no ponto de carga satura e nao esconde o resto
    vmax = max(m["s_max_fora"] for _, _, m, _ in res)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(1, len(res), figsize=(LARGURA_FIG_POL, 3.6),
                             gridspec_kw={"wspace": 0.08})
    for i, (ax, (nome, _, m, vm)) in enumerate(zip(axes, res)):
        im = ax.imshow(vm, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f"({'abcd'[i]}) {nome.replace(' ', chr(10), 1)}\n"
                     f"$\\sigma_{{PN}} = {_br(m['s_pn'], 3)}$", fontsize=_pt(11), pad=8)
        _add_holes(ax, bc)
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cb.set_label("Von Mises (adimensional)", fontsize=_pt(11))
    cb.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    fig_path = os.path.join(_OUTDIR, "15_tensao_deslocamento.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figura: {fig_path}")
