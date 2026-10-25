"""
analises pos-otimizacao do suporte automotivo 30x30:
  robustez : (a1) topologia nominal fixa reavaliada sob carga perturbada
             (magnitude +-20%, angulo +-15 graus, ponto +-1 elemento;
             material e analitico, c ~ 1/E); (a2) simp re-otimizado do
             zero sob perturbacao de geometria dos furos e de carga.
  dfam     : fracao de elementos solidos em violacao da regra de overhang
             de 45 graus (ibhadode et al. 2023; dangal e jung 2023) nas
             quatro orientacoes de impressao, suporte automotivo 30x30,
             incluindo a topologia ppo polida quando 11 ja tiver salvo o
             npz. verificacao pos-otimizacao; nao e termo do objetivo nem
             da recompensa.
             nota (2026-09-20): o cantilever foi removido desta tabela —
             usava simp_cantilever() em 60x30 e a topologia ppo polida em
             60x30 (09), domino que deixou de existir apos a comparacao
             20x10 vs 60x30 upscaled ser apontada como injusta. ver nota em
             09_resultados_cantilever.py. simp_cantilever() abaixo fica sem
             uso, mantida (nao apagada) por precaucao.
  malha    : re-otimizacao em 30x30, 60x60 e 90x90 com escala fisica do
             filtro constante (rmin 1.5/3.0/4.5) e geometria fisica
             constante (furos e olhais escalados em ponto flutuante em
             build_bcs) — teste canonico de independencia de malha do simp
             (sigmund 2001).

saidas: resultados/12_robustez.{png,csv}, 12_dfam.{png,csv}, 12_malha.png
uso: python 12_analises_automotivo.py [--so robustez|dfam|malha|tudo]
"""

import os
import csv
import time
import argparse
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
precompute_filter = _mod.precompute_filter
optimality_criteria_update = _mod.optimality_criteria_update

_s10 = spec_from_file_location("simp_auto", os.path.join(_DIR, "10_simp_automotivo.py"))
_m10 = module_from_spec(_s10)
_s10.loader.exec_module(_m10)
build_bcs = _m10.build_bcs
rodar_simp = _m10.rodar_simp

NELX, NELY = 30, 30
VOL_FRAC = 0.3
PENAL = 3.0
RMIN = 1.5

_OUTDIR = os.path.join(_DIR, "resultados")
os.makedirs(_OUTDIR, exist_ok=True)

# --- fonte casada com o corpo do texto -------------------------------------
# este script gera figuras com larguras de figsize diferentes (robustez,
# dfam e malha); por isso o fator de escala e calculado por figura via
# _pt_de(W), nao por uma constante global.
LARGURA_MANCHA_POL = 6.30     # 16 cm de mancha do USPSC (A4, margens 3 cm / 2 cm)
def _pt_de(W, p=12): return p * W / LARGURA_MANCHA_POL

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": _pt_de(18), "axes.titlesize": _pt_de(18), "axes.labelsize": _pt_de(18),
    "xtick.labelsize": _pt_de(18), "ytick.labelsize": _pt_de(18),
    "legend.fontsize": _pt_de(18), "figure.titlesize": _pt_de(18),
    "axes.linewidth": 1.5, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _br(v, nd=2):
    """numero com virgula decimal para mathtext ({,} evita o espaco apos a virgula)."""
    return f"{v:.{nd}f}".replace(".", "{,}")


# rotulos das figuras (as chaves continuam sem acento no console/csv)
_ROTULO = {
    "furos +1 em x": "furos $+1$ em $x$",
    "furo sup. +1 em y": "furo sup. $+1$ em $y$",
    "furo inf. -1 em y": "furo inf. $-1$ em $y$",
    "raio furo +1": "raio do furo $+1$",
    "carga +20%": "carga $+20\\%$",
    "angulo +15 graus": "ângulo $+15^\\circ$",
}


def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def _carrega_topologia_ppo(nome):
    """le a topologia polida salva por 09/11; None se ainda nao foi gerada."""
    p = os.path.join(_DIR, "resultados", "dados", f"topologia_ppo_{nome}.npz")
    if not os.path.exists(p):
        print(f"  aviso: {p} nao encontrado (rode 09/11 antes) — analise do ppo pulada")
        return None
    return np.load(p)["polido"]


def avaliar_topologia(x2d, bc, nelx=NELX, nely=NELY):
    """compliancia da topologia fixa sob as bcs dadas (um solve de mef).
    clip inferior evita K singular em topologias binarias com zeros exatos."""
    x2d = np.clip(x2d, 1e-3, 1.0)
    U = FE(nelx, nely, x2d, PENAL, fixed_dofs=bc["fixeddofs"], force_vector=bc["F"])
    c, _, _, _ = sensitivity_analysis(nelx, nely, x2d, PENAL, U)
    return c


def robustez():
    print("=" * 74)
    print(" ANALISE DE ROBUSTEZ DO SUPORTE AUTOMOTIVO 30x30")
    print("=" * 74)

    bc_nom = build_bcs(NELX, NELY)
    x_nom, _, it_nom = rodar_simp(bc_nom)
    # rodar_simp devolve x do passo k mas c do passo k-1 (c e calculado antes
    # do update do OC, dentro do loop); reavalia aqui no MESMO campo x_nom que
    # sera perturbado abaixo, para o denominador bater com a topologia real
    # (2026-09-22 — antes disso, a linha "carga +-20%" do SIMP saia com
    # +43.96%/-36.02% em vez do exato +44.00%/-36.00% que a identidade
    # C ~ F^2 exige, por causa desse descompasso de um passo).
    c_nom = avaliar_topologia(x_nom, bc_nom)
    print(f"\nnominal: C = {c_nom:.4f} ({it_nom} iters) [baseline do tcc ~22.92]")

    linhas = [("caso", "analise", "C", "dC_pct")]

    print("\nA1) peca projetada (topologia nominal fixa) sob carga perturbada:")
    casos_a1 = [
        ("carga +20%", dict(load_scale=1.2)),
        ("carga -20%", dict(load_scale=0.8)),
        ("angulo +15 graus", dict(load_angle_deg=+15.0)),
        ("angulo -15 graus", dict(load_angle_deg=-15.0)),
        ("aplicacao +1 elem", dict(load_shift=+1)),
        ("aplicacao -1 elem", dict(load_shift=-1)),
    ]
    for nome, kw in casos_a1:
        bc_p = build_bcs(NELX, NELY, **kw)
        c = avaliar_topologia(x_nom, bc_p)
        d = 100.0 * (c - c_nom) / c_nom
        print(f"   {nome:<20s} C = {c:8.4f}   dC = {d:+7.2f}%")
        linhas.append((nome, "A1_peca_fixa", f"{c:.4f}", f"{d:+.2f}"))
    print("   material: C escala exatamente com 1/E "
          "(E-20% -> C+25.00%; E+20% -> C-16.67%) — analitico")
    linhas.append(("E -20% (analitico)", "A1_peca_fixa", f"{c_nom/0.8:.4f}", "+25.00"))
    linhas.append(("E +20% (analitico)", "A1_peca_fixa", f"{c_nom/1.2:.4f}", "-16.67"))

    print("\nA2) projeto re-otimizado sob perturbacao (simp do zero):")
    (h1x, h1y), (h2x, h2y) = bc_nom["centers"]
    casos_a2 = [
        ("furos +1 em x", dict(centers=[(h1x + 1, h1y), (h2x + 1, h2y)])),
        ("furo sup. +1 em y", dict(centers=[(h1x, h1y + 1), (h2x, h2y)])),
        ("furo inf. -1 em y", dict(centers=[(h1x, h1y), (h2x, h2y - 1)])),
        ("raio furo +1", dict(r_hole=bc_nom["r_hole"] + 1)),
        ("carga +20%", dict(load_scale=1.2)),
        ("angulo +15 graus", dict(load_angle_deg=+15.0)),
    ]
    topos = [("nominal", x_nom, c_nom)]
    for nome, kw in casos_a2:
        bc_p = build_bcs(NELX, NELY, **kw)
        x_p, c_p, it_p = rodar_simp(bc_p)
        d = 100.0 * (c_p - c_nom) / c_nom
        print(f"   {nome:<20s} C = {c_p:8.4f}   dC = {d:+7.2f}%   ({it_p} iters)")
        linhas.append((nome, "A2_reotimizado", f"{c_p:.4f}", f"{d:+.2f}"))
        topos.append((nome, x_p, c_p))

    print("\nA3) monte carlo simplificado da carga (peca fixa, 50 amostras):")
    # distribuicoes ancoradas nos casos deterministicos de A1: os extremos
    # +-15 graus e +-20% correspondem a +-2 sigma, logo angulo ~ N(0, 7.5)
    # e magnitude ~ N(1, 0.10); um solve de mef por amostra, peca fixa
    rng = np.random.default_rng(0)
    n_mc = 50
    cs_mc = np.empty(n_mc)
    for i in range(n_mc):
        ang = float(rng.normal(0.0, 7.5))
        mag = float(max(rng.normal(1.0, 0.10), 0.01))
        bc_mc = build_bcs(NELX, NELY, load_angle_deg=ang, load_scale=mag)
        cs_mc[i] = avaliar_topologia(x_nom, bc_mc)
    p95 = float(np.percentile(cs_mc, 95))
    print(f"   C medio = {cs_mc.mean():.4f} | desvio = {cs_mc.std():.4f} | "
          f"p95 = {p95:.4f} | max = {cs_mc.max():.4f} "
          f"({100*(p95-c_nom)/c_nom:+.1f}% no p95 vs nominal)")
    linhas.append(("MC carga: C medio", "A3_monte_carlo", f"{cs_mc.mean():.4f}",
                   f"{100*(cs_mc.mean()-c_nom)/c_nom:+.2f}"))
    linhas.append(("MC carga: desvio", "A3_monte_carlo", f"{cs_mc.std():.4f}", ""))
    linhas.append(("MC carga: p95", "A3_monte_carlo", f"{p95:.4f}",
                   f"{100*(p95-c_nom)/c_nom:+.2f}"))

    x_ppo = _carrega_topologia_ppo("automotivo")
    if x_ppo is not None:
        c_ppo0 = avaliar_topologia(x_ppo, bc_nom)
        print(f"\nA1-PPO) topologia ppo polida fixa (C nominal = {c_ppo0:.4f}) "
              "sob as mesmas perturbacoes:")
        for nome, kw in casos_a1:
            bc_p = build_bcs(NELX, NELY, **kw)
            c = avaliar_topologia(x_ppo, bc_p)
            d = 100.0 * (c - c_ppo0) / c_ppo0
            print(f"   {nome:<20s} C = {c:8.4f}   dC = {d:+7.2f}%")
            linhas.append((nome, "A1_ppo_fixa", f"{c:.4f}", f"{d:+.2f}"))
        cs_mc_p = np.empty(n_mc)
        rng_p = np.random.default_rng(0)   # mesmas amostras do simp
        for i in range(n_mc):
            ang = float(rng_p.normal(0.0, 7.5))
            mag = float(max(rng_p.normal(1.0, 0.10), 0.01))
            bc_mc = build_bcs(NELX, NELY, load_angle_deg=ang, load_scale=mag)
            cs_mc_p[i] = avaliar_topologia(x_ppo, bc_mc)
        p95_p = float(np.percentile(cs_mc_p, 95))
        print(f"   MC carga: C medio = {cs_mc_p.mean():.4f} | desvio = "
              f"{cs_mc_p.std():.4f} | p95 = {p95_p:.4f} "
              f"({100*(p95_p-c_ppo0)/c_ppo0:+.1f}% no p95 vs nominal do ppo)")
        linhas.append(("MC carga: C medio", "A3_ppo", f"{cs_mc_p.mean():.4f}",
                       f"{100*(cs_mc_p.mean()-c_ppo0)/c_ppo0:+.2f}"))
        linhas.append(("MC carga: p95", "A3_ppo", f"{p95_p:.4f}",
                       f"{100*(p95_p-c_ppo0)/c_ppo0:+.2f}"))

    csv_path = os.path.join(_OUTDIR, "12_robustez.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"\ntabela: {csv_path}")

    # grade em varias linhas (2026-09-22): 7 paineis numa linha so nao cabe
    # com fonte de 12pt real (mesmo problema geometrico do 11, ver nota la —
    # a razao texto/painel e invariante quando so a largura total muda).
    # 4 colunas por linha resolve, mantendo a largura por painel razoavel.
    n_show = len(topos)
    n_cols = min(n_show, 4)
    n_linhas = -(-n_show // n_cols)  # ceil
    largura_robustez = 2.6 * n_cols
    fig, axes = plt.subplots(n_linhas, n_cols,
                             figsize=(largura_robustez, 3.2 * n_linhas),
                             gridspec_kw={"hspace": 0.4})
    axes = np.atleast_2d(axes).flatten()
    for ax, (nome, x2d, c) in zip(axes, topos):
        ax.imshow(1.0 - x2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{_ROTULO.get(nome, nome)}\n$C = {_br(c)}$",
                     fontsize=_pt_de(largura_robustez))
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)
    for ax in axes[n_show:]:
        ax.axis("off")
    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Sensibilidade do Projeto Ótimo a Perturbações de Geometria "
    #              "e Carga — Suporte $30\\times30$", y=1.04)
    fig_path = os.path.join(_OUTDIR, "12_robustez.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figura: {fig_path}")


def _peca_fabricada(x_design, bc_nom, bc_p, e_campo):
    """topologia fixa como sairia da fabricacao: o olhal nominal e impresso
    cheio e o furo e usinado na posicao/raio perturbados (bc_p). o campo de
    modulo e embutido na densidade, x_eff = x * E**(1/p), pois FE() usa
    E_e = x_e**p sem saturar x."""
    x = np.clip(x_design, 1e-3, 1.0).copy()
    x[bc_nom["void"] | bc_nom["solid"]] = 1.0
    x[bc_p["void"]] = 1e-3
    return x * e_campo ** (1.0 / PENAL)


def mc_multivariado(n_mc=500, semente=0):
    """monte carlo com perturbacao simultanea de direcao, magnitude e ponto
    de aplicacao da carga, posicao e raio dos furos e modulo do material,
    sobre a peca projetada fixa (um solve de mef por amostra). desvios
    ancorados nos casos deterministicos (extremos = +-2 sigma): angulo
    +-15 graus, magnitude e modulo global +-20%, ponto e furos +-1 elemento,
    raio +-0,5 elemento; heterogeneidade local do modulo com cv de 10% e
    correlacao espacial de ~2 elementos."""
    from scipy.ndimage import gaussian_filter
    print("=" * 74)
    print(f" MONTE CARLO MULTIVARIADO (peca fixa, {n_mc} amostras)")
    print("=" * 74)

    bc_nom = build_bcs(NELX, NELY)
    x_nom, _, _ = rodar_simp(bc_nom)
    topologias = [("SIMP", x_nom)]
    x_ppo = _carrega_topologia_ppo("automotivo")
    if x_ppo is not None:
        topologias.append(("PPO", x_ppo))

    rng = np.random.default_rng(semente)
    (h1x, h1y), (h2x, h2y) = bc_nom["centers"]
    # angulo e ponto entram em valor absoluto: o efeito deles em C e par
    # (+-15 graus mudam C no mesmo sentido) e a correlacao linear o zeraria.
    # furos_dx > 0 = furos em direcao a carga; aprox_furos > 0 = furos se
    # aproximam da linha media (y cresce para baixo nas figuras)
    nomes_var = ["|angulo|", "magnitude", "|ponto|", "furos_dx", "aprox_furos",
                 "raio", "E_global"]
    amostras = []
    for _ in range(n_mc):
        ang = rng.normal(0.0, 7.5)
        mag = max(rng.normal(1.0, 0.10), 0.01)
        ponto = int(np.rint(rng.normal(0.0, 0.5)))
        d = rng.normal(0.0, 0.5, size=4)
        raio = max(bc_nom["r_hole"] + rng.normal(0.0, 0.25), 0.5)
        e_glob = max(rng.normal(1.0, 0.10), 0.05)
        campo = gaussian_filter(rng.standard_normal((NELY, NELX)), 2.0, mode="nearest")
        campo = campo / campo.std() * 0.10
        e_campo = e_glob * np.exp(campo - 0.5 * 0.10 ** 2)
        bc_p = build_bcs(NELX, NELY,
                         centers=[(h1x + d[0], h1y + d[1]), (h2x + d[2], h2y + d[3])],
                         r_hole=raio, load_angle_deg=ang, load_scale=mag,
                         load_shift=ponto)
        amostras.append(((ang, mag, ponto, 0.5 * (d[0] + d[2]),
                          0.5 * (d[1] - d[3]), raio, e_glob), bc_p, e_campo))
    X = np.array([a[0] for a in amostras])
    X[:, 0] = np.abs(X[:, 0])
    X[:, 2] = np.abs(X[:, 2])

    linhas = [("topologia", "estatistica", "C", "dC_pct")]
    resultados = {}
    for nome, x_des in topologias:
        c0 = avaliar_topologia(x_des, bc_nom)
        cs = np.array([avaliar_topologia(_peca_fabricada(x_des, bc_nom, bc_p, e), bc_p)
                       for _, bc_p, e in amostras])
        rel = 100.0 * (cs - c0) / c0
        p5, p50, p95 = np.percentile(cs, [5, 50, 95])
        print(f"\n{nome}: C nominal = {c0:.4f}")
        print(f"   media = {cs.mean():.4f} ({rel.mean():+.1f}%) | desvio = "
              f"{cs.std(ddof=1):.4f} (cv {100*cs.std(ddof=1)/cs.mean():.1f}%)")
        print(f"   p5 = {p5:.4f} | mediana = {p50:.4f} | p95 = {p95:.4f} "
              f"({100*(p95-c0)/c0:+.1f}%) | max = {cs.max():.4f} "
              f"({rel.max():+.1f}%)")
        corr = [np.corrcoef(X[:, j], cs)[0, 1] for j in range(X.shape[1])]
        print("   correlacao de cada variavel com C: " + ", ".join(
            f"{v} {r:+.2f}" for v, r in zip(nomes_var, corr)))
        for est, val in [("nominal", c0), ("media", cs.mean()),
                         ("desvio", cs.std(ddof=1)), ("p5", p5), ("mediana", p50),
                         ("p95", p95), ("max", cs.max())]:
            dc = "" if est == "desvio" else f"{100*(val-c0)/c0:+.2f}"
            linhas.append((nome, est, f"{val:.4f}", dc))
        for v, r in zip(nomes_var, corr):
            linhas.append((nome, f"corr_{v}", f"{r:+.3f}", ""))

        # C*E/F^2 remove a parte algebrica (C ~ F^2/E); sobra o efeito de
        # direcao/ponto da carga, furos e heterogeneidade local do modulo
        cn = cs * X[:, 6] / X[:, 1] ** 2
        reln = 100.0 * (cn - c0) / c0
        q5, q95 = np.percentile(cn, [5, 95])
        print(f"   normalizada C*E/F^2: media = {cn.mean():.4f} ({reln.mean():+.1f}%) | "
              f"desvio = {cn.std(ddof=1):.4f} (cv {100*cn.std(ddof=1)/cn.mean():.1f}%) | "
              f"p5 = {q5:.4f} ({100*(q5-c0)/c0:+.1f}%) | p95 = {q95:.4f} "
              f"({100*(q95-c0)/c0:+.1f}%)")
        corr_n = [np.corrcoef(X[:, j], cn)[0, 1] for j in (0, 2, 3, 4, 5)]
        print("   correlacao com a normalizada: " + ", ".join(
            f"{nomes_var[j]} {r:+.2f}" for j, r in zip((0, 2, 3, 4, 5), corr_n)))
        for est, val in [("norm_media", cn.mean()), ("norm_desvio", cn.std(ddof=1)),
                         ("norm_p5", q5), ("norm_p95", q95)]:
            dc = "" if est == "norm_desvio" else f"{100*(val-c0)/c0:+.2f}"
            linhas.append((nome, est, f"{val:.4f}", dc))
        resultados[nome] = (rel, reln)

    csv_path = os.path.join(_OUTDIR, "12_mc_multivariado.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"\ntabela: {csv_path}")

    largura = 9.0
    fs = _pt_de(largura)
    fig, axes = plt.subplots(1, 2, figsize=(largura, 3.6), gridspec_kw={"wspace": 0.3})
    titulos = ["(a) Compliância $C$", "(b) Normalizada $C\\,E/F^2$"]
    for k, ax in enumerate(axes):
        for (nome, par), cor in zip(resultados.items(), ["0.35", "tab:red"]):
            ax.hist(par[k], bins=30, alpha=0.6, color=cor, label=nome)
        ax.axvline(0.0, color="black", lw=1.0)
        ax.set_title(titulos[k], fontsize=fs)
        ax.set_xlabel("Variação em relação\nao nominal (%)", fontsize=fs)
        ax.tick_params(labelsize=fs)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
        _spines(ax)
    axes[0].set_ylabel("Amostras", fontsize=fs)
    axes[0].legend(fontsize=fs, frameon=False)
    fig_path = os.path.join(_OUTDIR, "12_mc_multivariado.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figura: {fig_path}")


def simp_cantilever(nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=3.0,
                    max_iters=200):
    """simp classico do caso de estudo (carga no meio da aresta direita);
    c esperado ~72.85."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    load_node = (nely + 1) * nelx + nely // 2
    F[2 * load_node + 1] = -1.0
    fixed_nodes = np.arange(0, nely + 1)
    fixeddofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])

    H, Hs = precompute_filter(nelx, nely, rmin)
    x = np.full(nelx * nely, volfrac)
    change, loop, c = 1.0, 0, 0.0
    while change > 0.01 and loop < max_iters:
        loop += 1
        xold = x.copy()
        x2d = x.reshape(nely, nelx, order="F")
        U = FE(nelx, nely, x2d, penal, fixed_dofs=fixeddofs, force_vector=F)
        c, dc, _, _ = sensitivity_analysis(nelx, nely, x2d, penal, U)
        x = optimality_criteria_update(nelx, nely, x, volfrac, dc, H, Hs)
        change = np.linalg.norm(x - xold, np.inf)
    return x.reshape(nely, nelx, order="F"), c, loop


# direcao de impressao -> (sinal de avanco da base, eixo percorrido);
# referencial das figuras do tcc: linha 0 = topo da imagem
_DIRECOES = {
    "base embaixo (constrói p/ cima)": (+1, "linha"),
    "base em cima (constrói p/ baixo)": (-1, "linha"),
    "base à esquerda (constrói p/ dir.)": (+1, "coluna"),
    "base à direita (constrói p/ esq.)": (-1, "coluna"),
}


def violacao_overhang(x2d, direcao, thresh=0.5):
    """fracao de elementos solidos sem suporte a 45 graus na direcao dada.
    suportado = na base, ou com vizinho solido na camada anterior dentro do
    cone de 45 graus (offset -1, 0 ou +1 na transversal)."""
    solid = x2d > thresh
    sgn, eixo = _DIRECOES[direcao]
    S = solid if eixo == "linha" else solid.T
    nrow = S.shape[0]
    ordem = range(nrow - 1, -1, -1) if sgn == +1 else range(nrow)
    viol = np.zeros_like(S, dtype=bool)
    prev = None
    for r in ordem:
        linha = S[r]
        if prev is None:
            prev = linha
            continue
        sup = prev.copy()
        sup[1:] |= prev[:-1]
        sup[:-1] |= prev[1:]
        viol[r] = linha & ~sup
        prev = linha
    if eixo == "coluna":
        viol = viol.T
    n_solid = int(solid.sum())
    return float(viol.sum()) / max(n_solid, 1), viol


def dfam():
    print("=" * 74)
    print(" ANALISE DfAM: VIOLACAO DE OVERHANG A 45 GRAUS")
    print("=" * 74)

    print("\nregenerando topologias (deterministico)...")
    # cantilever removido (2026-09-20): ver nota no topo do arquivo.
    # x_cant, c_cant, it_c = simp_cantilever()
    # print(f"  cantilever 60x30: C = {c_cant:.4f} ({it_c} iters) [esperado ~72.85]")
    bc = build_bcs(NELX, NELY)
    x_auto, c_auto, it_a = rodar_simp(bc)
    print(f"  automotivo 30x30: C = {c_auto:.4f} ({it_a} iters) [esperado ~22.93]")

    casos = [("Suporte $30\\times30$ (SIMP)", x_auto)]
    # cantilever removido (2026-09-20): ver nota no topo do arquivo.
    # x_pc = _carrega_topologia_ppo("cantilever")
    # if x_pc is not None:
    #     casos.append(("cantilever 60x30 (PPO polido)", x_pc))
    x_pa = _carrega_topologia_ppo("automotivo")
    if x_pa is not None:
        casos.append(("Suporte $30\\times30$ (PPO polido)", x_pa))

    linhas = [("topologia", "direcao_impressao", "violacao_pct")]
    resultados = {}
    for nome, x2d in casos:
        print(f"\n{nome}:")
        melhores = []
        for direcao in _DIRECOES:
            frac, viol = violacao_overhang(x2d, direcao)
            pct = 100.0 * frac
            print(f"   {direcao:<38s} violacao = {pct:5.1f}%")
            linhas.append((nome, direcao, f"{pct:.1f}"))
            melhores.append((pct, direcao, viol))
        melhores.sort()
        resultados[nome] = (x2d, melhores)
        print(f"   -> melhor orientacao: {melhores[0][1]} "
              f"({melhores[0][0]:.1f}% de suporte sacrificial)")

    csv_path = os.path.join(_OUTDIR, "12_dfam.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"\ntabela: {csv_path}")

    # titulos encurtados (2026-09-22): "melhor/pior orientacao: base a
    # esquerda (constroi p/ dir.)" nao cabe em 3 colunas com fonte de 12pt
    # real (mesmo problema geometrico do 11/robustez); a parte entre
    # parenteses (justificativa da direcao) fica so no console/csv.
    n_rows = len(resultados)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.2 * n_rows),
                             gridspec_kw={"wspace": 0.2, "hspace": 0.55})
    axes = np.atleast_2d(axes)
    for row, (nome, (x2d, melhores)) in enumerate(resultados.items()):
        ax = axes[row, 0]
        ax.imshow(1.0 - x2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(nome, fontsize=_pt_de(18))
        for ax, (pct, direcao, viol), tag in [
                (axes[row, 1], melhores[0], "Melhor"),
                (axes[row, 2], melhores[-1], "Pior")]:
            base = np.where(x2d > 0.5, 0.75, 0.0)
            ax.imshow(1.0 - base, cmap="gray", vmin=0, vmax=1)
            yy, xx = np.nonzero(viol)
            ax.scatter(xx, yy, s=6, c="red", marker="s")
            direcao_curta = direcao.split(" (")[0]
            ax.set_title(f"{tag}: {direcao_curta}\n${_br(pct, 1)}\\%$",
                         fontsize=_pt_de(18))
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            _spines(ax)
    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Análise de Fabricabilidade (DfAM) — Elementos em Violação "
    #              "da Regra de Overhang de 45$^\\circ$", y=1.01)
    fig_path = os.path.join(_OUTDIR, "12_dfam.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figura: {fig_path}")


def malha():
    print("=" * 74)
    print(" VERIFICACAO COM MALHAS REFINADAS")
    print("=" * 74)
    malhas = [(30, 30, 1.5), (60, 60, 3.0), (90, 90, 4.5)]
    res = []
    for nelx, nely, rmin in malhas:
        bc = build_bcs(nelx, nely)
        t0 = time.time()
        x2d, c, loop = rodar_simp(bc, nelx=nelx, nely=nely, rmin=rmin,
                                  max_iters=300)
        dt = time.time() - t0
        vf = float(x2d.mean())
        frac, _ = violacao_overhang(x2d, "base à esquerda (constrói p/ dir.)")
        print(f"  {nelx}x{nely} (rmin={rmin}): C = {c:9.4f} | {loop:3d} iters "
              f"| VF = {vf:.3f} | overhang(base esq.) = {100*frac:4.1f}% "
              f"| {dt:5.1f}s")
        res.append((nelx, nely, rmin, x2d, c, loop, frac))

    largura_malha = 3.4 * len(res)
    fig, axes = plt.subplots(1, len(res), figsize=(largura_malha, 3.6))
    for ax, (nelx, nely, rmin, x2d, c, loop, frac) in zip(axes, res):
        ax.imshow(1.0 - x2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"${nelx}\\times{nely}$, $r_{{min}}={_br(rmin, 1)}$\n"
                     f"$C = {_br(c)}$ | {loop} iterações", fontsize=_pt_de(largura_malha))
        ax.set_xticks([])
        ax.set_yticks([])
        _spines(ax)
    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Verificação de Independência de Malha — Suporte "
    #              "com Escala Física de Filtro Constante", y=1.03)
    fig_path = os.path.join(_OUTDIR, "12_malha.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nfigura: {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=str, default="tudo",
                    choices=["robustez", "mc", "dfam", "malha", "tudo"])
    args = ap.parse_args()
    if args.so in ("robustez", "tudo"):
        robustez()
    if args.so in ("mc", "tudo"):
        mc_multivariado()
    if args.so in ("dfam", "tudo"):
        dfam()
    if args.so in ("malha", "tudo"):
        malha()
