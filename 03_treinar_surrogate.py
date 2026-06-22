"""
Treino e benchmark dos modelos substitutos (RF, GP+PCA, CNN).

Pipeline em duas partes ativas:
    Secao 1 — treina os tres modelos e gera as curvas de aprendizado.
    Secao 3 — mede a latencia de inferencia dos substitutos contra o MEF.
(A secao 2, de amostragem ativa, foi descontinuada; ver nota no main.)

Reprodutibilidade: a divisao treino/teste usa GroupShuffleSplit por run_id
com SEED fixo. Executar novamente reproduz as mesmas figuras.

Entrada:
    resultados/dados/topo_data_24x24.npz (gerado por 01_coletar_dados_surrogate.py)
    campos: x, y, fx, fy, load_node, run_id, nelx, nely

Saidas:
    resultados/03_learning_curves_surrogate.png
    resultados/03_comparacao_surrogates.png
    resultados/dados/surrogate_{cnn,rf,gp}_24x24.{pth,pkl}
"""

import os
import sys
import pickle
import time
from importlib.util import spec_from_file_location, module_from_spec

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

# os modulos 00 e 02 comecam com digito e nao podem ser importados pelo nome;
# carregamos pelos caminhos dos arquivos
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "00_simp_e_fem.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FEMSolver = _mod.FEMSolver

_spec2 = spec_from_file_location("arq_cnn", os.path.join(_DIR, "02_arquitetura_cnn_surrogate.py"))
_mod2 = module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)
SurrogateCNN = _mod2.SurrogateCNN

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.linewidth": 1.5, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SEED = 42                # semente fixa para reproduzir os splits e as figuras
C_CUTOFF = 320.0         # teto de compliancia operacional (alinhado ao codigo 01)
TEST_SPLIT = 0.2         # divisao treino/teste 80/20
N_GP_MAX = 500           # limite de amostras do GP (custo O(n^3); Rasmussen e Williams)
N_PCA_COMP = 10          # componentes principais para o GP (escolha empirica)
N_FEM_TIMING = 50        # amostras para medir a latencia do MEF

CORES = {"RF": "#CC6600", "GP+PCA": "#7733AA", "CNN": "#004488"}
MARKS = {"RF": "o", "GP+PCA": "s", "CNN": "^"}


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------

def _spines(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("black")
        sp.set_linewidth(1.5)


def build_force_maps(load_nodes, fxs, fys, nely, nelx):
    """Constroi mapas espaciais 2D de Fx e Fy para cada amostra.

    O mapa e zerado exceto no elemento adjacente ao no de carga. Codificar a
    carga como canais espaciais segue Banga et al. (2018).

    Numeracao dos nos: node = (nely+1)*ix + iy; elemento adjacente
    (elx, ely) = (min(ix, nelx-1), min(iy, nely-1)).
    """
    n = len(load_nodes)
    fx_maps = np.zeros((n, nely, nelx), dtype=np.float32)
    fy_maps = np.zeros((n, nely, nelx), dtype=np.float32)
    for i, (node, fx, fy) in enumerate(zip(load_nodes, fxs, fys)):
        ix = node // (nely + 1)
        iy = node % (nely + 1)
        elx = min(ix, nelx - 1)
        ely = min(iy, nely - 1)
        fx_maps[i, ely, elx] = fx
        fy_maps[i, ely, elx] = fy
    return fx_maps, fy_maps


def compliance_fem(density, solver):
    """Compliancia via SIMP modificado (Andreassen et al., 2011).

    Usada apenas para medir a latencia do MEF na secao 3; o valor retornado
    nao alimenta as figuras de acuracia.
    """
    from scipy.ndimage import convolve
    _, ce, _ = solver.solve(density)
    x_phys = np.clip(convolve(density, solver.H, mode="reflect"), 0.001, 1.0)
    E0, Emin = 1.0, 1e-9
    return float(np.sum((Emin + x_phys**3.0 * (E0 - Emin)) * ce))


# ---------------------------------------------------------------------------
# Treino dos modelos
# ---------------------------------------------------------------------------

def train_cnn(x_train, y_train, crit, epochs=200):
    """Treina a CNN com Adam e parada antecipada.

    Reserva 10% do treino como validacao interna (sem tocar no conjunto de
    teste) e interrompe se a perda de validacao nao melhora por 30 epocas.
    Entrada: (N, 3, nely, nelx); canais densidade, Fx, Fy (Banga et al., 2018).
    """
    xt, xv, yt, yv = train_test_split(x_train, y_train, test_size=0.1, random_state=SEED)
    model = SurrogateCNN()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=min(64, len(xt)), shuffle=True)
    best_v, pat, best_w = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(xv), yv).item()
        if val_loss < best_v:
            best_v = val_loss
            best_w = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 30:
            break
    model.load_state_dict(best_w)
    return model


def train_gp(X_train_flat, Y_train_log, n_pca):
    """Treina o GP sobre as componentes principais da entrada.

    A entrada e [densidade(576) + fx + fy] = 578 features na malha 24x24,
    reduzida por PCA. O GP e limitado a N_GP_MAX amostras pelo custo O(n^3).
    """
    scaler = StandardScaler()
    Xt_sc = scaler.fit_transform(X_train_flat)
    n_comp = max(2, min(n_pca, X_train_flat.shape[0] - 1, X_train_flat.shape[1]))
    pca = PCA(n_components=n_comp, random_state=SEED)
    Xt_pca = pca.fit_transform(Xt_sc)
    n_gp = min(N_GP_MAX, len(Xt_pca))
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(1e-3, (1e-5, 1e1))
    # alpha funciona como nugget: amostras adjacentes do SIMP sao quase colineares
    # e deixam a matriz de covariancia quase singular sem essa regularizacao
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-2,
                                  n_restarts_optimizer=3, random_state=SEED)
    gp.fit(Xt_pca[:n_gp], Y_train_log[:n_gp])
    return gp, scaler, pca, n_gp


def train_gp_evaluate(X_train_flat, Y_train_log, X_test_flat, Y_test_real):
    """Treina o GP e avalia o MAE no conjunto de teste (usado na secao 2)."""
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train_flat)
    Xte_sc = scaler.transform(X_test_flat)
    n_comp = max(2, min(N_PCA_COMP, Xtr_sc.shape[0] - 1, Xtr_sc.shape[1]))
    pca = PCA(n_components=n_comp)
    Xtr_pca = pca.fit_transform(Xtr_sc)
    Xte_pca = pca.transform(Xte_sc)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e4)) + WhiteKernel(1e-3, (1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-2,
                                  n_restarts_optimizer=1, random_state=SEED)
    gp.fit(Xtr_pca, Y_train_log)
    Y_pred, Y_std = gp.predict(Xte_pca, return_std=True)
    mae = mean_absolute_error(Y_test_real, np.expm1(Y_pred))
    return mae, gp, scaler, pca


def selecionar_maior_incerteza(gp, scaler, pca, X_pool_flat, ja_selecionados):
    """Seleciona a amostra de maior variancia de predicao do GP.

    Amostragem adaptativa por variancia de predicao (Kudela e Matousek, 2022).
    """
    restantes = [i for i in range(len(X_pool_flat)) if i not in ja_selecionados]
    if not restantes:
        return None, -1
    X_cand = X_pool_flat[restantes]
    Xc_pca = pca.transform(scaler.transform(X_cand))
    _, stds = gp.predict(Xc_pca, return_std=True)
    melhor = restantes[np.argmax(stds)]
    return melhor, stds.max()


# ---------------------------------------------------------------------------
# Secao 1 — treino e curvas de aprendizado
# ---------------------------------------------------------------------------

def secao1_treino(xt_flat, xe_flat, yt_log, ye_log, ye_lin,
                  xt_cnn, xe_cnn, yt_cnn, ye_cnn,
                  out_lc, out_cnn, out_rf, out_gp):
    crit = nn.MSELoss()
    resultados = {}

    # random forest (baseline)
    print("\n[S1] treinando RF...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
    rf.fit(xt_flat, yt_log)
    trf = time.time() - t0
    t0 = time.time()
    p_rf = np.expm1(rf.predict(xe_flat))
    inf_rf = (time.time() - t0) / len(xe_flat) * 1e6
    mae_rf = mean_absolute_error(ye_lin, p_rf)
    r2_rf = r2_score(ye_lin, p_rf)
    resultados["RF"] = {"mae": mae_rf, "r2": r2_rf, "inf": inf_rf,
                        "pred": p_rf, "train_s": trf}
    print(f"  RF -> MAE: {mae_rf:.2f} | R2: {r2_rf:.4f} | inf: {inf_rf:.1f}us")

    # gp com pca
    print(f"\n[S1] treinando GP+PCA (limite {N_GP_MAX} amostras)...")
    t0 = time.time()
    gp, scaler, pca, n_usado = train_gp(xt_flat, yt_log, N_PCA_COMP)
    tgp = time.time() - t0
    print(f"  variancia explicada ({N_PCA_COMP} PCs): {pca.explained_variance_ratio_.sum():.1%}")
    xe_pca = pca.transform(scaler.transform(xe_flat))
    t0 = time.time()
    p_gp = np.expm1(gp.predict(xe_pca))
    inf_gp = (time.time() - t0) / len(xe_flat) * 1e6
    mae_gp = mean_absolute_error(ye_lin, p_gp)
    r2_gp = r2_score(ye_lin, p_gp)
    resultados["GP+PCA"] = {"mae": mae_gp, "r2": r2_gp, "inf": inf_gp,
                            "pred": p_gp, "train_s": tgp,
                            "n_gp": n_usado, "n_total": len(xt_flat)}
    print(f"  GP -> MAE: {mae_gp:.2f} | R2: {r2_gp:.4f} | inf: {inf_gp:.1f}us")

    # cnn (parada antecipada na validacao interna, sem tocar no teste)
    print("\n[S1] treinando CNN...")
    t0 = time.time()
    model = train_cnn(xt_cnn, yt_cnn, crit)
    tcnn = time.time() - t0
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        p_cnn = np.expm1(model(xe_cnn).squeeze().numpy())
    inf_cnn = (time.time() - t0) / len(xe_cnn) * 1e6
    mae_cnn = mean_absolute_error(ye_lin, p_cnn)
    r2_cnn = r2_score(ye_lin, p_cnn)
    resultados["CNN"] = {"mae": mae_cnn, "r2": r2_cnn, "inf": inf_cnn,
                         "pred": p_cnn, "train_s": tcnn}
    print(f"  CNN -> MAE: {mae_cnn:.2f} | R2: {r2_cnn:.4f} | inf: {inf_cnn:.1f}us")

    # salva os modelos para o benchmark da secao 3
    torch.save(model.state_dict(), out_cnn)
    with open(out_rf, "wb") as f:
        pickle.dump(rf, f)
    with open(out_gp, "wb") as f:
        pickle.dump({"gp": gp, "scaler": scaler, "pca": pca}, f)
    print("  pesos salvos.")

    # curvas de aprendizado (MAE vs numero de amostras)
    out_lc_csv = out_lc.replace(".png", "_dados.csv")
    n_treinos = [50, 100, 200, 400, 800, 1200, len(xt_flat)]
    n_treinos = [n for n in n_treinos if n <= len(xt_flat)]
    lc_res = {"RF": [], "GP+PCA": [], "CNN": []}

    if os.path.exists(out_lc_csv):
        print(f"\n[S1] carregando learning curves do CSV: {out_lc_csv}")
        import pandas as pd
        lc_df = pd.read_csv(out_lc_csv)
        n_treinos = lc_df["n"].tolist()
        lc_res = {c: lc_df[c].tolist() for c in ["RF", "GP+PCA", "CNN"]}
    else:
        print("\n[S1] gerando learning curves...")
        for n in n_treinos:
            print(f"  N={n}...", end=" ", flush=True)
            xt_f_sub = xt_flat[:n]
            yt_sub = yt_log[:n]

            rf_lc = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
            rf_lc.fit(xt_f_sub, yt_sub)
            lc_res["RF"].append(mean_absolute_error(ye_lin, np.expm1(rf_lc.predict(xe_flat))))

            gp_lc, sc_lc, pca_lc, _ = train_gp(xt_f_sub, yt_sub, N_PCA_COMP)
            xte_pca_lc = pca_lc.transform(sc_lc.transform(xe_flat))
            lc_res["GP+PCA"].append(mean_absolute_error(ye_lin, np.expm1(gp_lc.predict(xte_pca_lc))))

            cnn_lc = train_cnn(xt_cnn[:n], yt_cnn[:n], crit)
            cnn_lc.eval()
            with torch.no_grad():
                yp_cnn_lc = np.expm1(cnn_lc(xe_cnn).squeeze().numpy())
            lc_res["CNN"].append(mean_absolute_error(ye_lin, yp_cnn_lc))
            print(f"RF={lc_res['RF'][-1]:.1f} | GP={lc_res['GP+PCA'][-1]:.1f} | CNN={lc_res['CNN'][-1]:.1f}")

        import pandas as pd
        pd.DataFrame({"n": n_treinos, **lc_res}).to_csv(out_lc_csv, index=False)
        print(f"  dados salvos: {out_lc_csv}")

    fig_lc, ax_lc = plt.subplots(figsize=(9, 5))
    for nome, maes in lc_res.items():
        ax_lc.plot(n_treinos, maes, MARKS[nome] + "-", color=CORES[nome], lw=2, ms=8, label=nome)
    ax_lc.axvline(N_GP_MAX, color=CORES["GP+PCA"], ls=":", lw=1.5, alpha=0.7)
    ax_lc.text(N_GP_MAX * 1.05, ax_lc.get_ylim()[0] if ax_lc.get_ylim()[0] > 0 else 0.5,
               f"limite GP\n(N={N_GP_MAX})", color=CORES["GP+PCA"], fontsize=9, va="bottom")
    ax_lc.set_xscale("log")
    ax_lc.set_xlabel("num. de amostras de treino")
    ax_lc.set_ylabel("MAE")
    ax_lc.set_title("Curvas de Aprendizado dos Modelos Substitutos — Cantilever 24×24")
    ax_lc.legend()
    _spines(ax_lc)
    fig_lc.tight_layout()
    fig_lc.savefig(out_lc)
    plt.close(fig_lc)
    print(f"  salvo: {out_lc}")

    return resultados, model, rf, gp, scaler, pca


# ---------------------------------------------------------------------------
# Secao 2 — amostragem ativa vs aleatoria (descontinuada; ver nota no main)
# ---------------------------------------------------------------------------

def secao2_active_sampling(X_op_flat, Y_op, X_test_flat, Y_test, out_fig):
    n0 = 100
    n_max = 500
    eval_step = 25

    rng = np.random.default_rng(SEED)
    step_ns = list(range(n0, n_max + 1, eval_step))

    # mesma semente inicial nas duas linhas para que partam do mesmo ponto
    idx_op_shared = rng.permutation(len(X_op_flat))
    X_seed = X_op_flat[idx_op_shared[:n0]].copy()
    Y_seed = np.log1p(Y_op[idx_op_shared[:n0]])

    print("\n[S2] amostragem aleatoria...")
    x_rand, y_rand, i_rand = X_seed.copy(), Y_seed.copy(), n0
    random_maes, random_ns = [], []
    for n_target in step_ns:
        mae, *_ = train_gp_evaluate(x_rand, y_rand, X_test_flat, Y_test)
        random_maes.append(mae)
        random_ns.append(n_target)
        print(f"  n={n_target} -> MAE={mae:.2f}")
        if n_target < n_max:
            n_add = min(eval_step, len(idx_op_shared) - i_rand)
            if n_add > 0:
                x_rand = np.vstack([x_rand, X_op_flat[idx_op_shared[i_rand:i_rand+n_add]]])
                y_rand = np.append(y_rand, np.log1p(Y_op[idx_op_shared[i_rand:i_rand+n_add]]))
                i_rand += n_add

    print("\n[S2] amostragem ativa (GP-guided)...")
    x_active = X_seed.copy()
    y_active = Y_seed.copy()
    selecionados = set(idx_op_shared[:n0].tolist())
    active_maes, active_ns = [], []
    for n_target in step_ns:
        mae, gp_m, sc_m, pca_m = train_gp_evaluate(x_active, y_active, X_test_flat, Y_test)
        active_maes.append(mae)
        active_ns.append(n_target)
        print(f"  n={n_target} -> MAE={mae:.2f}")
        if n_target < n_max:
            # selecao em lote: nao reestima a covariancia a cada ponto, mas e rapido
            for _ in range(eval_step):
                idx_sel, _ = selecionar_maior_incerteza(gp_m, sc_m, pca_m, X_op_flat, selecionados)
                if idx_sel is None:
                    break
                selecionados.add(idx_sel)
                x_active = np.vstack([x_active, X_op_flat[idx_sel].reshape(1, -1)])
                y_active = np.append(y_active, np.log1p(Y_op[idx_sel]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(random_ns, random_maes, "o-", color="#004488", lw=2, ms=8, label="Aleatoria")
    ax.plot(active_ns, active_maes, "s--", color="#BB5566", lw=2, ms=8, label="Ativa (GP)")
    ax.set_xlabel("N amostras de treino")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.set_title(f"Ativa vs Aleatória (GP+PCA)\nMalha 24×24 | C≤{C_CUTOFF:.0f}")
    _spines(ax)
    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"  salvo: {out_fig}")

    delta = (random_maes[-1] - active_maes[-1]) / random_maes[-1] * 100
    print(f"\n  aleatoria @ n={n_max}: MAE={random_maes[-1]:.3f}")
    print(f"  ativa(gp) @ n={n_max}: MAE={active_maes[-1]:.3f}")
    print(f"  diff: {delta:+.1f}% (negativo = ativa foi pior)")


# ---------------------------------------------------------------------------
# Secao 3 — benchmark de latencia substituto vs MEF
# ---------------------------------------------------------------------------

def secao3_benchmark(X_test_dens, X_test_flat, Y_test, X_test_cnn,
                     model_cnn, rf, gp_bundle, solver, out_fig):
    gp = gp_bundle["gp"]
    scaler = gp_bundle["scaler"]
    pca = gp_bundle["pca"]
    X_test_pca = pca.transform(scaler.transform(X_test_flat))

    print(f"\n[S3] medindo FEM ({N_FEM_TIMING} amostras)...")
    compliance_fem(X_test_dens[0], solver)  # warmup, descarta o overhead da primeira chamada
    t_fem_v = np.zeros(N_FEM_TIMING)
    for i, dens in enumerate(X_test_dens[:N_FEM_TIMING]):
        t0 = time.perf_counter()
        compliance_fem(dens, solver)
        t_fem_v[i] = time.perf_counter() - t0
    t_fem_us = np.mean(t_fem_v) * 1e6
    print(f"  FEM: {t_fem_us/1000:.3f} ms/amostra ({t_fem_us:.0f} us)")

    print("\n[S3] medindo surrogates...")
    surrogate_fns = {
        "RF":     lambda: np.expm1(rf.predict(X_test_flat)),
        "GP+PCA": lambda: np.expm1(gp.predict(X_test_pca)),
        "CNN":    lambda: np.expm1(model_cnn(X_test_cnn).detach().squeeze().numpy()),
    }
    results = {}
    for name, predict_fn in surrogate_fns.items():
        _ = predict_fn()  # warmup
        t0 = time.perf_counter()
        Y_pred = predict_fn()
        t_us = (time.perf_counter() - t0) / len(X_test_flat) * 1e6
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)
        speedup = t_fem_us / t_us
        results[name] = dict(pred=Y_pred, mae=mae, r2=r2, t_us=t_us, speedup=speedup)
        print(f"  {name:8s}: MAE={mae:.2f} | R2={r2:.4f} | Inf={t_us:.1f}us | Speedup={speedup:.0f}x")

    print(f"\n[S3] resumo completo (FEM baseline = {t_fem_us/1000:.3f} ms/amostra):")
    print(f"  {'Modelo':<10} {'Inferência':>14} {'MAE':>8} {'R²':>8} {'Speedup':>10}")
    print(f"  {'FEM':<10} {t_fem_us/1000:.3f} ms/am  {'—':>8} {'—':>8} {'1x':>10}")
    for n in results:
        r = results[n]
        print(f"  {n:<10} {r['t_us']:.1f} µs/am  {r['mae']:>8.2f} {r['r2']:>8.4f} {r['speedup']:>9.0f}x")

    vmin, vmax = Y_test.min(), Y_test.max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, ["RF", "GP+PCA", "CNN"]):
        res = results[name]
        pred_plot = np.clip(res["pred"], vmin * 0.5, vmax * 1.5)
        ax.scatter(Y_test, pred_plot, alpha=0.35, s=12, color=CORES[name], zorder=2)
        label_ideal = f'MAE={res["mae"]:.1f}  |  R²={res["r2"]:.4f}\n{res["t_us"]:.1f} µs/am'
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label=label_ideal, zorder=3)
        ax.set_xlim(vmin * 0.9, vmax * 1.1)
        ax.set_ylim(vmin * 0.9, vmax * 1.1)
        ax.set_xlabel("Compliance real")
        ax.set_ylabel("Compliance predito")
        ax.set_title(name, pad=8)
        ax.legend(fontsize=8.5, loc="upper left")
        _spines(ax)

    fig.suptitle("Desempenho Preditivo dos Modelos Substitutos — Conjunto de Teste", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"  salvo: {out_fig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    _HERE = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_HERE, "resultados", "dados", "topo_data_24x24.npz")

    if not os.path.exists(data_path):
        print(f"[ERRO] dataset nao encontrado: {data_path}")
        print("       rode 01_coletar_dados_surrogate.py primeiro.")
        sys.exit(1)

    out_lc = os.path.join(_HERE, "resultados", "03_learning_curves_surrogate.png")
    out_bm = os.path.join(_HERE, "resultados", "03_comparacao_surrogates.png")
    out_cnn = os.path.join(_HERE, "resultados", "dados", "surrogate_cnn_24x24.pth")
    out_rf = os.path.join(_HERE, "resultados", "dados", "surrogate_rf_24x24.pkl")
    out_gp = os.path.join(_HERE, "resultados", "dados", "surrogate_gp_24x24.pkl")
    os.makedirs(os.path.dirname(out_lc), exist_ok=True)
    os.makedirs(os.path.dirname(out_cnn), exist_ok=True)

    print("carregando dados...")
    data = np.load(data_path)
    x_raw_all = data["x"]
    y_lin_all = data["y"]
    fx_all = data["fx"]
    fy_all = data["fy"]
    nodes_all = data["load_node"]
    nelx = int(data["nelx"])
    nely = int(data["nely"])

    # filtra a regiao operacional antes do split, para o teste ser representativo
    mask = y_lin_all <= C_CUTOFF
    x_raw = x_raw_all[mask]
    y_lin = y_lin_all[mask]
    fx = fx_all[mask]
    fy = fy_all[mask]
    nodes = nodes_all[mask]
    # log1p comprime a cauda assimetrica da distribuicao de compliancia
    y_log = np.log1p(y_lin)
    n_total = len(x_raw)
    print(f"regiao operacional C<={C_CUTOFF:.0f}: {n_total} amostras")
    print(f"C min: {y_lin.min():.2f} | C max: {y_lin.max():.2f} | std: {y_lin.std():.2f}")

    # GroupShuffleSplit por run: um split aleatorio por amostra colocaria iteracoes
    # adjacentes do mesmo run no treino e no teste, inflando o R2 por memorizacao
    run_id_all = data["run_id"] if "run_id" in data else np.arange(n_total)
    run_ids = run_id_all[mask]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)
    idx = np.arange(n_total)
    id_tr, id_te = next(gss.split(idx, groups=run_ids))
    print(f"treino: {len(id_tr)} | teste fixo: {len(id_te)} (GroupShuffleSplit por run)")

    # mapas de forca reconstruidos a partir de load_node (economia de disco)
    fx_maps, fy_maps = build_force_maps(nodes, fx, fy, nely, nelx)

    # RF / GP: vetor [densidade(576) + fx + fy] = 578 features (malha 24x24)
    x_flat_dens = x_raw.reshape(n_total, -1)
    x_flat_full = np.hstack([x_flat_dens, fx.reshape(-1, 1), fy.reshape(-1, 1)]).astype(np.float32)
    xt_flat = x_flat_full[id_tr]
    xe_flat = x_flat_full[id_te]
    yt_log = y_log[id_tr]
    ye_log = y_log[id_te]
    ye_lin = y_lin[id_te]

    # CNN: tensor de 3 canais (densidade, Fx_map, Fy_map) — Banga et al. (2018)
    x_cnn_all = np.stack([x_raw, fx_maps, fy_maps], axis=1).astype(np.float32)
    xt_cnn = torch.FloatTensor(x_cnn_all[id_tr])
    xe_cnn = torch.FloatTensor(x_cnn_all[id_te])
    yt_cnn = torch.FloatTensor(yt_log).unsqueeze(1)
    ye_cnn = torch.FloatTensor(ye_log).unsqueeze(1)

    # solver para o timing do MEF na secao 3 (BCs padrao de cantilever)
    solver = FEMSolver(nelx, nely, vol_frac=0.5, penal=3.0, rmin=1.5)
    mid = nely // 2
    node_mid = (nely + 1) * nelx + mid
    solver.F = np.zeros(solver.ndof)
    solver.F[2 * node_mid + 1] = -1.0

    resultados, model_cnn, rf, gp, scaler, pca = secao1_treino(
        xt_flat, xe_flat, yt_log, ye_log, ye_lin,
        xt_cnn, xe_cnn, yt_cnn, ye_cnn,
        out_lc, out_cnn, out_rf, out_gp,
    )

    # Secao 2 (amostragem ativa) descontinuada: empiricamente, a amostragem ativa
    # por GP nao superou a amostragem sistematica das iteracoes do SIMP, entao a
    # figura foi removida do trabalho. As funcoes ficam disponiveis para reuso.

    gp_bundle = {"gp": gp, "scaler": scaler, "pca": pca}
    secao3_benchmark(
        x_raw[id_te], xe_flat, ye_lin, xe_cnn,
        model_cnn, rf, gp_bundle, solver, out_bm,
    )

    print("\n=== resumo final ===")
    for nome, r in resultados.items():
        nota = f" [GP cap: {r.get('n_gp')}/{len(id_tr)}]" if nome == "GP+PCA" else ""
        print(f"  {nome:8s}: MAE={r['mae']:.2f} | R2={r['r2']:.4f} | "
              f"inf={r['inf']:.1f}us | treino={r.get('train_s', 0):.0f}s{nota}")