"""
comparacao final no suporte automotivo 30x30, todas com as mesmas bcs (10)
e binarizacao a vf exato: simp binarizado, baseline guloso eso (mesmo
espaco de acoes do agente, ver rodar_guloso em 05), agente ppo bruto e
ppo + polimento simp (15 iteracoes, woldseth et al. 2022). figura de 3
paineis horizontais (simp, ppo bruto, von mises do ppo bruto); o polimento
continua calculado e salvo no .npz para uso do 12_analises_automotivo.py,
mas nao aparece mais nesta figura (2026-09-22).

requer o checkpoint resultados/dados/v21_automotivo_30x30/best_model.zip
(gerado por 08_treinar_ppo.py --problem automotive).
saidas: resultados/11_comparacao_automotivo.png,
        resultados/dados/topologia_ppo_automotivo.npz (consumido pelo 12)
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FE = _mod.FE
sensitivity_analysis = _mod.sensitivity_analysis
precompute_filter = _mod.precompute_filter

_s05 = spec_from_file_location("rl_env_mask", os.path.join(_DIR, "05_ambiente_rl.py"))
_m05 = module_from_spec(_s05)
_s05.loader.exec_module(_m05)
TopoOptMaskEnv = _m05.TopoOptMaskEnv
topo_mask_fn = _m05.topo_mask_fn
rodar_guloso = _m05.rodar_guloso

# os checkpoints referenciam a politica pelo nome de modulo "politica_fcn";
# registrar antes do exec para o maskableppo.load desserializar
_s06 = spec_from_file_location("politica_fcn", os.path.join(_DIR, "06_politica_fcn.py"))
_m06 = module_from_spec(_s06)
sys.modules["politica_fcn"] = _m06
_s06.loader.exec_module(_m06)

_s10 = spec_from_file_location("simp_auto", os.path.join(_DIR, "10_simp_automotivo.py"))
_m10 = module_from_spec(_s10)
_s10.loader.exec_module(_m10)
build_bcs = _m10.build_bcs
oc_passive = _m10.oc_passive
rodar_simp = _m10.rodar_simp

NELX, NELY, VOL_FRAC = 30, 30, 0.3
PENAL, RMIN, EMIN = 3.0, 1.5, 1e-3
POLISH_ITERS = 15
FILTRO_ESO = 1.5   # raio do professor guloso medido como o melhor em 30x30

_MODEL_PADRAO = os.path.join(_DIR, "resultados", "dados",
                             "v21_automotivo_30x30", "best_model.zip")
_OUTDIR = os.path.join(_DIR, "resultados")
os.makedirs(_OUTDIR, exist_ok=True)

# --- fonte casada com o corpo do texto -------------------------------------
# nota (2026-09-22): a largura original do figsize (22") pediria _pt(12) =
# 41.9pt — um titulo de 2 linhas nesse tamanho estoura o painel em 4 colunas.
# So encolher a figura NAO resolve: a razao texto/painel e invariante quando
# fonte e largura escalam juntas (texto ~ 12*(W/6.3), painel ~ W/4, razao
# constante independente de W). A correcao real e mudar a grade de 1x4 para
# 2x2, que dobra a largura disponivel por painel.
LARGURA_FIG_POL = 9.0         # largura do figsize deste script (grade 2x2, ver nota)
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


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def binarizar_vf(x2d, bc, vol_frac=VOL_FRAC):
    """binariza a vf exato: olhal solido, furo vazio, e os elementos livres
    mais densos preenchem o volume restante."""
    n = x2d.size
    alvo_solidos = int(round(vol_frac * n))
    xb = np.full_like(x2d, EMIN)
    xb[bc["solid"]] = 1.0
    xb[bc["void"]] = EMIN
    livre = ~(bc["solid"] | bc["void"])
    resta = alvo_solidos - int(bc["solid"].sum())
    dens_livres = np.where(livre, x2d, -1.0).flatten()
    ordem = np.argsort(dens_livres)[::-1][:max(resta, 0)]
    xb_flat = xb.flatten()
    xb_flat[ordem] = 1.0
    return xb_flat.reshape(x2d.shape)


def avaliar(x2d, bc):
    U = FE(NELX, NELY, x2d, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
    c, _, _, vm = sensitivity_analysis(NELX, NELY, x2d, PENAL, U)
    return c, vm


def polir_simp(x0_2d, bc, n_iters=POLISH_ITERS):
    """polimento simp partindo da topologia do agente (passivos preservados)."""
    void_flat = bc["void"].flatten(order="F")
    solid_flat = bc["solid"].flatten(order="F")
    H, Hs = precompute_filter(NELX, NELY, RMIN)
    x = np.clip(x0_2d.flatten(order="F").astype(float), EMIN, 1.0)
    x[solid_flat] = 1.0
    x[void_flat] = EMIN
    t0 = time.time()
    for _ in range(n_iters):
        x2d = x.reshape(NELY, NELX, order="F")
        U = FE(NELX, NELY, x2d, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
        _, dc, _, _ = sensitivity_analysis(NELX, NELY, x2d, PENAL, U)
        x = oc_passive(x, VOL_FRAC, dc, H, Hs, solid_flat, void_flat)
    return x.reshape(NELY, NELX, order="F"), time.time() - t0


def episodio_greedy(model_path, seed=0):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    env = TopoOptMaskEnv(nelx=NELX, nely=NELY, vol_frac=VOL_FRAC,
                         problem_type="automotive")
    wenv = ActionMasker(env, topo_mask_fn)
    model = MaskablePPO.load(model_path)
    obs, _ = wenv.reset(seed=seed)
    done, info, t0 = False, {}, time.time()
    while not done:
        masks = wenv.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, term, trunc, info = wenv.step(int(action))
        done = term or trunc
    return (env.density.copy(), info.get("compliance", float("nan")),
            info.get("deadlock", False), env.n_removed, time.time() - t0)


def varredura_sementes(c_ref, seed=0):
    """avalia checkpoints de sementes extras (v21_automotivo_30x30_seed*/)
    e imprime media e desvio do C_greedy — estabilidade do treino."""
    import glob
    pads = sorted(glob.glob(os.path.join(
        _DIR, "resultados", "dados", "v21_automotivo_30x30_seed*",
        "best_model.zip")))
    if not pads:
        return
    print("\nestabilidade entre sementes:")
    print(f"   seed 42 (canonico): C = {c_ref:.4f}")
    cs = [c_ref]
    for p in pads:
        tag = os.path.basename(os.path.dirname(p))
        _, c, _, _, _ = episodio_greedy(p, seed)
        print(f"   {tag}: C = {c:.4f}")
        cs.append(c)
    cs = np.array(cs)
    print(f"   media = {cs.mean():.4f} | desvio amostral = {cs.std(ddof=1):.4f} "
          f"| min = {cs.min():.4f} | max = {cs.max():.4f} | n = {len(cs)}")


def _br(v, nd=2):
    """numero com virgula decimal para mathtext ({,} evita o espaco apos a virgula)."""
    return f"{v:.{nd}f}".replace(".", "{,}")


def _add_holes(ax, bc):
    for (hx, hy) in bc["centers"]:
        ax.add_patch(Circle((hx - 0.5, hy - 0.5), bc["r_hole"],
                            color="red", fill=False, lw=1.2))
        ax.add_patch(Circle((hx - 0.5, hy - 0.5), bc["r_boss"],
                            color="green", fill=False, lw=1.2, ls="--"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=_MODEL_PADRAO)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 74)
    print(" COMPARACAO FINAL — SUPORTE AUTOMOTIVO 30x30")
    print("=" * 74)
    bc = build_bcs(NELX, NELY)

    t0 = time.time()
    x_simp, c_simp_cont, it_simp = rodar_simp(bc)
    t_simp = time.time() - t0
    xb_simp = binarizar_vf(x_simp, bc)
    c_simp_bin, _ = avaliar(xb_simp, bc)
    print(f"\nSIMP puro:    C_cont = {c_simp_cont:.4f} ({it_simp} iters, "
          f"{t_simp:.1f}s) | C_bin = {c_simp_bin:.4f}")

    # baseline guloso eso (uma linha, sem treino) no mesmo espaco de acoes
    x_eso, c_eso, n_eso, t_eso = rodar_guloso(
        nelx=NELX, nely=NELY, vol_frac=VOL_FRAC, problem_type="automotive",
        raio_filtro=FILTRO_ESO, seed=args.seed, verbose=False)
    print(f"ESO guloso:   C = {c_eso:.4f} (filtro rmin={FILTRO_ESO}) | "
          f"{n_eso} remocoes | {t_eso:.1f}s")

    if not os.path.exists(args.model):
        print(f"\nerro: checkpoint nao encontrado: {args.model}")
        print("rode 08_treinar_ppo.py --problem automotive antes.")
        sys.exit(1)

    x_ppo, c_ppo, deadlock, n_rem, t_ep = episodio_greedy(args.model, args.seed)
    flag = "  [DEADLOCK antes do VF alvo]" if deadlock else ""
    print(f"PPO bruto:    C = {c_ppo:.4f} | {n_rem} remocoes | {t_ep:.1f}s{flag}")

    x_pol, t_pol = polir_simp(x_ppo, bc)
    xb_pol = binarizar_vf(x_pol, bc)
    c_pol_bin, vm_pol = avaliar(xb_pol, bc)

    # topologia final para as analises de robustez/dfam do 12
    _dados = os.path.join(_DIR, "resultados", "dados")
    os.makedirs(_dados, exist_ok=True)
    np.savez_compressed(os.path.join(_dados, "topologia_ppo_automotivo.npz"),
                        bruto=x_ppo, polido=xb_pol)
    dC = c_pol_bin - c_simp_bin
    print(f"PPO + polim.: C_bin = {c_pol_bin:.4f} ({POLISH_ITERS} iters, "
          f"{t_pol:.1f}s) | dC vs SIMP_bin = {dC:+.4f} "
          f"({100*dC/c_simp_bin:+.1f}%)")

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, nc_ppo = scipy.ndimage.label(x_ppo > 0.5, structure=struct)

    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {'Metodo':<28} {'C':>10}  {'iters MEF':>10}")
    print(sep)
    print(f"  {'SIMP binarizado':<28} {c_simp_bin:>10.2f}  {it_simp:>10d}")
    print(f"  {'ESO guloso (sem treino)':<28} {c_eso:>10.2f}  {n_eso:>10d}")
    print(f"  {'PPO bruto':<28} {c_ppo:>10.2f}  {n_rem:>10d}")
    print(f"  {'PPO + polimento (bin)':<28} {c_pol_bin:>10.2f}  {POLISH_ITERS:>10d}")
    print(sep)
    print(f"  componentes conexos do PPO bruto: {nc_ppo}")
    # fracao de volume final de cada solucao (dominio inteiro, furos contam
    # como vazio), para mostrar que todas comparam a mesma massa
    print(f"  VF final: SIMP_cont {x_simp.mean():.4f} | SIMP_bin {xb_simp.mean():.4f}"
          f" | ESO {(x_eso > 0.5).mean():.4f} | PPO bruto {(x_ppo > 0.5).mean():.4f} | PPO polido {xb_pol.mean():.4f}")

    varredura_sementes(c_ppo, args.seed)

    # figura reduzida a 3 paineis horizontais, sem o polimento SIMP (pedido
    # do usuario, 2026-09-22): (a) SIMP puro, (b) PPO bruto, (c) Von Mises do
    # PPO bruto (nao do polido). o polimento em si (xb_pol/c_pol_bin/vm_pol)
    # continua sendo calculado acima e salvo no .npz, pois 12_analises_automotivo.py
    # ainda consome a chave "polido" para as analises de robustez/dfam.
    # clip para EMIN (nao 0.0 puro): FE() usa penalizacao SIMP sem piso Emin
    # (x^p puro, ver 01_fem_simp.py), entao elementos com densidade exatamente
    # zero deixam a matriz de rigidez singular (acusado por "dgstrf info 1"
    # do solver esparso) e o campo de Von Mises resultante sai todo NaN/zero.
    xb_ppo = np.clip((x_ppo > 0.5).astype(float), EMIN, 1.0)
    _, vm_ppo = avaliar(xb_ppo, bc)
    vm_ppo_show = vm_ppo.copy()
    vm_ppo_show[bc["void"]] = 0.0

    # titulos sem cor (2026-09-22): texto sempre preto, sem destaque colorido
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), gridspec_kw={"wspace": 0.15})
    paineis = [
        (1.0 - xb_simp, "gray",
         f"(a) SIMP puro\nbinarizado\n$C_{{bin}} = {_br(c_simp_bin)}$ | {it_simp} iterações"),
        (1.0 - xb_ppo, "gray",
         f"(b) PPO com\nMascaramento de Ações\n$C = {_br(c_ppo)}$ | {n_rem} remoções"),
        (vm_ppo_show / max(vm_ppo_show.max(), 1e-12), "turbo",
         "(c) Von Mises\ndo PPO (aprox.)"),
    ]
    for ax, (img, cmap, titulo) in zip(axes, paineis):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=_pt(11), color="black", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)
        if cmap == "gray":
            _add_holes(ax, bc)
    fig_path = os.path.join(_OUTDIR, "11_comparacao_automotivo.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nfigura: {fig_path}")
