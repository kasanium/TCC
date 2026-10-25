"""
arquitetura da cnn substituta (entrada de 3 canais: densidade, fx, fy —
banga et al. 2018; pilha 32/64/128 — kumarn et al. 2025) e treino/benchmark
dos tres substitutos (rf, gp+pca, cnn) sobre o dataset do 02. split
treino/teste por run_id (groupshufflesplit, sem vazamento entre iteracoes
da mesma execucao). alvo: log1p(compliancia).
saidas: resultados/03_learning_curves_surrogate.png,
resultados/03_comparacao_surrogates.png,
resultados/dados/surrogate_{cnn,rf,gp}_24x24.*
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
_spec = spec_from_file_location("simp_e_fem", os.path.join(_DIR, "01_fem_simp.py"))
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FEMSolver = _mod.FEMSolver


# --- fonte casada com o corpo do texto -------------------------------------
# este script tem duas figuras de larguras diferentes (curvas de aprendizado
# 9 pol.; comparacao dos substitutos, empilhada em 1 coluna, 6.3 pol.), entao
# usamos um _pt_de(W) por figura em vez de uma unica LARGURA_FIG_POL global
# (mesmo ajuste feito em 09_resultados_cantilever.py).
LARGURA_MANCHA_POL = 6.30     # 16 cm de mancha do USPSC (A4, margens 3 cm / 2 cm)
def _pt_de(W, p=12): return p * (W / LARGURA_MANCHA_POL)  # pt na pagina -> pt no matplotlib
_pt = lambda p: _pt_de(9.0, p)  # default: curvas de aprendizado (figsize largura 9)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": _pt(12), "axes.titlesize": _pt(12), "axes.labelsize": _pt(12),
    "xtick.labelsize": _pt(12), "ytick.labelsize": _pt(12),
    "legend.fontsize": _pt(12), "figure.titlesize": _pt(12),
    "axes.linewidth": 1.5, "figure.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _softplus_fc(in_features, out_features):
    """Camada linear seguida de Softplus (ativacao suave para a regressao)."""
    return nn.Sequential(nn.Linear(in_features, out_features), nn.Softplus())


class SurrogateCNN(nn.Module):
    """CNN para regressao de compliancia em malha 24x24.

    Dimensoes ao longo da rede (entrada 24x24):
        Conv(3->32, 3x3, pad=1):    24x24 -> 24x24
        MaxPool(2x2):               24x24 -> 12x12
        Conv(32->64, 3x3, pad=1):   12x12 -> 12x12
        Conv(64->128, 3x3, pad=1):  12x12 -> 12x12
        MaxPool(2x2):               12x12 -> 6x6
        Flatten:                    128 * 6 * 6 = 4608
    """

    _FLAT_DIM = 128 * 6 * 6  # 4608

    def __init__(self):
        super().__init__()

        # extrator convolucional 32/64/128 (Kumarn et al., 2025)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # regressao escalar -> log1p(compliancia)
        self.regressor = nn.Sequential(
            _softplus_fc(self._FLAT_DIM, 256),
            _softplus_fc(256, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)



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
               f"limite GP\n(N={N_GP_MAX})", color=CORES["GP+PCA"], fontsize=_pt(12), va="bottom")
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

    # empilhados em 1 coluna (2026-09-22): com 3 paineis lado a lado, cada
    # um so tem LARGURA_MANCHA_POL/3 = 2.1 pol. de pagina - insuficiente
    # para os rotulos dos eixos ("100"/"200") e a caixa de legenda em fonte
    # real de 12pt (mesmo problema geometrico de 11/12_robustez: a razao
    # texto/painel independe do figsize, so depende do numero de colunas).
    vmin, vmax = Y_test.min(), Y_test.max()
    _pt3 = lambda p: _pt_de(LARGURA_MANCHA_POL, p)
    fig, axes = plt.subplots(3, 1, figsize=(LARGURA_MANCHA_POL, 4.6 * 3))
    for ax, name in zip(axes, ["RF", "GP+PCA", "CNN"]):
        res = results[name]
        pred_plot = np.clip(res["pred"], vmin * 0.5, vmax * 1.5)
        ax.scatter(Y_test, pred_plot, alpha=0.35, s=12, color=CORES[name], zorder=2)
        label_ideal = (f'MAE = {res["mae"]:.1f}  |  R² = {res["r2"]:.4f}\n'
                       f'{res["t_us"]:.1f} µs por amostra').replace(".", ",")
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label=label_ideal, zorder=3)
        ax.set_xlim(vmin * 0.9, vmax * 1.1)
        ax.set_ylim(vmin * 0.9, vmax * 1.1)
        ax.set_xlabel("Compliância real", fontsize=_pt3(12))
        ax.set_ylabel("Compliância predita", fontsize=_pt3(12))
        ax.tick_params(labelsize=_pt3(12))
        ax.set_title(name, pad=8, fontsize=_pt3(12))
        ax.legend(fontsize=_pt3(12), loc="upper left")
        _spines(ax)

    # titulo geral comentado (2026-09-21): redundante com a legenda do LaTeX.
    # fig.suptitle("Desempenho Preditivo dos Modelos Substitutos — Conjunto de Teste", fontsize=13)
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
        print("       rode 02_dados_surrogate.py primeiro.")
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

    # amostragem ativa por gp foi testada e nao superou a amostragem
    # sistematica das iteracoes do simp; descontinuada

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