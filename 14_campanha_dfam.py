"""
campanha com penalidade de overhang ativa (suporte 30x30): compara o agente
de referencia (seed 42, sem penalidade) com o agente treinado com
--dfam_peso 10 --dfam_base baixo. mede a perda de rigidez (compliancia) contra
o ganho de fabricabilidade (fracao de elementos em violacao da regra de 45
graus nas quatro direcoes de construcao, mesma metrica de 12).

saidas: resultados/14_campanha_dfam.{csv,png}
uso: python 14_campanha_dfam.py
"""

import os
import csv
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from importlib.util import spec_from_file_location, module_from_spec

_DIR = os.path.dirname(os.path.abspath(__file__))


def _carrega(nome, arquivo):
    spec = spec_from_file_location(nome, os.path.join(_DIR, arquivo))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m11 = _carrega("res_auto", "11_resultados_automotivo.py")
m12 = _carrega("analises_auto", "12_analises_automotivo.py")

_DADOS = os.path.join(_DIR, "resultados", "dados")
MODELOS = [
    ("PPO de referência", os.path.join(_DADOS, "v21_automotivo_30x30", "best_model.zip")),
    ("PPO com penalidade ($w=10$)",
     os.path.join(_DADOS, "v21_automotivo_30x30_dfam10", "best_model.zip")),
]
BASE = "base embaixo (constrói p/ cima)"


def _br(v, nd=2):
    return f"{v:.{nd}f}".replace(".", "{,}")


def main():
    bc = m11.build_bcs(m11.NELX, m11.NELY)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    linhas = [("modelo", "C", "dC_pct", "VF", "componentes", "deadlock")
              + tuple(f"viol_{d.split(' (')[0]}" for d in m12._DIRECOES)]
    res = []
    for nome, path in MODELOS:
        x, c, deadlock, n_rem, _ = m11.episodio_greedy(path)
        solid = x > 0.5
        _, ncomp = scipy.ndimage.label(solid, structure=struct)
        viols = {d: 100.0 * m12.violacao_overhang(x, d)[0] for d in m12._DIRECOES}
        res.append((nome, x, c, viols))
        print(f"{nome}: C = {c:.4f} | VF = {solid.mean():.4f} | {n_rem} remocoes | "
              f"{ncomp} comp. | deadlock = {deadlock}")
        for d, v in viols.items():
            print(f"   {d:<38s} violacao = {v:5.1f}%")
        linhas.append([nome, f"{c:.4f}", "", f"{solid.mean():.4f}", ncomp, deadlock]
                      + [f"{v:.1f}" for v in viols.values()])

    c_ref = res[0][2]
    for linha, (_, _, c, _) in zip(linhas[1:], res):
        linha[2] = f"{100 * (c - c_ref) / c_ref:+.2f}"
    _, _, c_pen, v_pen = res[1]
    v_ref = res[0][3]
    print(f"\ntroca: compliancia {100*(c_pen-c_ref)/c_ref:+.1f}% | violacao com {BASE}: "
          f"{v_ref[BASE]:.1f}% -> {v_pen[BASE]:.1f}%")

    csv_path = os.path.join(m12._OUTDIR, "14_campanha_dfam.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"tabela: {csv_path}")

    largura = 9.0
    fs = m12._pt_de(largura)
    fig, axes = plt.subplots(1, len(res), figsize=(largura, 4.4), gridspec_kw={"wspace": 0.15})
    for ax, (nome, x, c, viols) in zip(axes, res):
        _, viol = m12.violacao_overhang(x, BASE)
        base = np.where(x > 0.5, 0.75, 0.0)
        ax.imshow(1.0 - base, cmap="gray", vmin=0, vmax=1)
        yy, xx = np.nonzero(viol)
        ax.scatter(xx, yy, s=6, c="red", marker="s")
        ax.set_title(f"{nome}\n$C = {_br(c)}$ | violação ${_br(viols[BASE], 1)}\\%$",
                     fontsize=fs)
        ax.set_xticks([])
        ax.set_yticks([])
        m12._spines(ax)
    fig_path = os.path.join(m12._OUTDIR, "14_campanha_dfam.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"figura: {fig_path}")


if __name__ == "__main__":
    main()
