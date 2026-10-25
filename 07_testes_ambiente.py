"""
testes do ambiente (05): equivalencia exata da mascara por pontos de
articulacao contra a forca bruta (scipy.ndimage.label), cache por estado,
terminacao limpa por deadlock, inercia da restricao de espessura minima
(regressao do diagnostico que a deixou desligada), simetria horizontal
(densidade simetrica e conexa em todo passo + mascara verificada por forca
bruta), penalidade dfam (peso 0 inerte, deslocamento exato do retorno),
fumaca da politica fcn (06) e fumaca de treino do 08. os dois
ultimos exigem torch/sb3-contrib e sao pulados com aviso se faltarem.
uso: python 07_testes_ambiente.py
"""

import os
import sys
import time
import numpy as np
import scipy.ndimage

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from importlib.util import spec_from_file_location, module_from_spec

_s = spec_from_file_location("rl_env_mask", os.path.join(_DIR, "05_ambiente_rl.py"))
_m = module_from_spec(_s)
_s.loader.exec_module(_m)
TopoOptMaskEnv = _m.TopoOptMaskEnv


def _random_state(env, rng, n_remove):
    """Leva o env a um estado conexo aleatorio removendo via mascara (sem
    reavaliar o MEF a cada passo — so a topologia importa para este teste)."""
    env.reset(seed=int(rng.integers(1e9)))
    for _ in range(n_remove):
        mask = env.get_action_mask()
        valid = np.nonzero(mask)[0]
        if len(valid) == 0:
            break
        a = int(rng.choice(valid))
        env.density[a // env.nelx, a % env.nelx] = 1e-4
        env.n_removed += 1
        env._mask_dirty = True
    return env


def test_equivalencia(n_trials=10):
    rng = np.random.default_rng(123)
    total = 0
    for thin_threshold in (0.0, 0.45):
        for (ptype, nx, ny, vf) in [("cantilever", 20, 10, 0.5),
                                    ("automotive", 30, 30, 0.3)]:
            for _ in range(n_trials):
                env = TopoOptMaskEnv(nelx=nx, nely=ny, vol_frac=vf, problem_type=ptype,
                                     thin_threshold=thin_threshold)
                n_rem = int(rng.integers(0, env.n_to_remove))
                env = _random_state(env, rng, n_rem)
                m_fast = env.get_action_mask()
                m_ref = env._get_action_mask_bruteforce()
                assert np.array_equal(m_fast, m_ref), \
                    (f"mascaras divergem: {ptype} {nx}x{ny} apos {n_rem} remocoes "
                     f"(thin_threshold={thin_threshold})")
                total += 1
    print(f"[OK] equivalencia articulacao == forca bruta ({total} estados, "
          f"com e sem restricao de espessura)")


def _rollout_aleatorio(nx, ny, vf, ptype, thin_threshold, seed):
    env = TopoOptMaskEnv(nelx=nx, nely=ny, vol_frac=vf, problem_type=ptype,
                         thin_threshold=thin_threshold)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    done, info = False, {}
    while not done:
        mask = env.get_action_mask()
        valid = np.nonzero(mask)[0]
        if len(valid) == 0:
            break
        a = int(rng.choice(valid))
        obs, r, done, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs)), "obs com NaN/inf"
        assert np.isfinite(r), "reward com NaN/inf"
    return info.get("compliance", float("nan")), info.get("deadlock", False)


def test_espessura_minima():
    """Documenta (e protege contra regressao de) o diagnostico v21: sob
    politica aleatoria e MESMA semente, thin_threshold in {0.0, 0.30, 0.45}
    devem produzir a MESMA compliancia final (a restricao nunca vincula
    nessa faixa), e 0.60 deve vincular e produzir deadlock. Ver item 3 da
    docstring de 05_ambiente_rl.py — a restricao fica desligada por
    padrao exatamente por causa disto."""
    for (ptype, nx, ny, vf) in [("cantilever", 20, 10, 0.5),
                                ("automotive", 30, 30, 0.3)]:
        c_base, dl_base = _rollout_aleatorio(nx, ny, vf, ptype, 0.0, seed=0)
        for th in (0.30, 0.45):
            c, dl = _rollout_aleatorio(nx, ny, vf, ptype, th, seed=0)
            assert c == c_base and dl == dl_base, (
                f"thin_threshold={th} mudou a trajetoria em {ptype} {nx}x{ny} "
                f"(C={c:.2f} vs base {c_base:.2f}) — a restricao deixou de "
                f"estar inerte nessa faixa; reavaliar o valor default (0.0) "
                f"e a docstring de 05_ambiente_rl.py, item 3")
        c_bind, dl_bind = _rollout_aleatorio(nx, ny, vf, ptype, 0.60, seed=0)
        assert dl_bind, (
            f"thin_threshold=0.60 nao vinculou em {ptype} {nx}x{ny} "
            f"(esperava deadlock) — o nucleo ou a malha mudaram; refazer a "
            f"medicao antes de reativar a restricao com outro valor")
        print(f"[OK] espessura minima em {ptype} {nx}x{ny}: inerte em "
              f"0.00-0.45 (C={c_base:.2f}), vincula e trava em 0.60")


def test_simetria_horizontal(n_episodios=8):
    """v22: sob simetria='horizontal' no cantilever, episodios aleatorios
    completos mantem densidade simetrica e um unico componente conexo em
    TODO passo (nao so no final), e a mascara bate contra uma verificacao
    independente (label apos remover o par elemento+espelho, por forca
    bruta). Tambem confere que o assert de BCs simetricas dispara para o
    automotivo (furos assimetricos por arredondamento, ver 05b, item de
    simetria)."""
    try:
        TopoOptMaskEnv(nelx=30, nely=30, vol_frac=0.3, problem_type="automotive",
                       simetria="horizontal")
        raise AssertionError("deveria ter disparado AssertionError para automotivo")
    except AssertionError as e:
        assert "simetria" in str(e), f"assert errado disparou: {e}"

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    n_ok = 0
    for ep in range(n_episodios):
        env = TopoOptMaskEnv(nelx=20, nely=10, vol_frac=0.5, problem_type="cantilever",
                             simetria="horizontal")
        rng = np.random.default_rng(ep)
        env.reset(seed=ep)
        done, info = False, {}
        while not done:
            mask = env.get_action_mask()
            valid = np.nonzero(mask)[0]
            if valid.size == 0:
                break
            a = int(rng.choice(valid))
            obs, r, done, trunc, info = env.step(a)
            solid = env.density > 0.5
            assert np.array_equal(solid, solid[::-1, :]), \
                f"quebrou simetria no episodio {ep}, passo {env.steps}"
            _, ncomp = scipy.ndimage.label(solid, structure=struct)
            assert ncomp == 1, f"desconectou no episodio {ep} (ncomp={ncomp})"
            assert np.all(np.isfinite(obs)) and np.isfinite(r), "obs/reward com NaN/inf"
        if not info.get("deadlock", False) and env.n_removed >= env.n_to_remove:
            n_ok += 1
    assert n_ok == n_episodios, (
        f"so {n_ok}/{n_episodios} episodios completaram sob simetria e politica aleatoria")

    # equivalencia da mascara contra verificacao independente, num estado
    # a meio caminho de um episodio
    env = TopoOptMaskEnv(nelx=20, nely=10, vol_frac=0.5, problem_type="cantilever",
                         simetria="horizontal")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        valid = np.nonzero(env.get_action_mask())[0]
        env.step(int(rng.choice(valid)))
    mask = env.get_action_mask()
    solid = env.density > 0.5
    erros = 0
    for i in range(env.n_elements):
        ely, elx = i // env.nelx, i % env.nelx
        j = int(env._mirror_flat[i])
        jy, jx = j // env.nelx, j % env.nelx
        if not solid[ely, elx] or env.protected_mask[ely, elx]:
            esperado = False
        elif j != i and (not solid[jy, jx] or env.protected_mask[jy, jx]):
            esperado = False
        else:
            test = solid.copy()
            test[ely, elx] = False
            if j != i:
                test[jy, jx] = False
            _, ncomp = scipy.ndimage.label(test, structure=struct)
            esperado = (ncomp == 1)
        erros += int(bool(mask[i]) != esperado)
    assert erros == 0, f"{erros} discrepancias na mascara sob simetria"
    print(f"[OK] simetria horizontal: {n_ok}/{n_episodios} episodios completos, "
          f"densidade simetrica e conexa em todo passo, mascara verificada "
          f"({env.n_elements} candidatos, 0 discrepancias)")


def test_dfam():
    """penalidade de overhang: com peso 0 nada muda; com peso > 0 a MESMA
    trajetoria (mesma semente; a mascara nao depende do dfam) tem retorno
    deslocado por exatamente -peso*(viol_final - viol_inicial)."""
    def _rollout(peso, seed=3):
        env = TopoOptMaskEnv(nelx=12, nely=6, vol_frac=0.5,
                             problem_type="cantilever",
                             dfam_peso=peso, dfam_base="baixo")
        rng = np.random.default_rng(seed)
        env.reset(seed=seed)
        viol0 = env._violacao_overhang(env.density > 0.5)
        total, acoes, done, info = 0.0, [], False, {}
        while not done:
            valid = np.nonzero(env.get_action_mask())[0]
            if valid.size == 0:
                break
            a = int(rng.choice(valid))
            acoes.append(a)
            _, r, done, _, info = env.step(a)
            total += r
        violf = env._violacao_overhang(env.density > 0.5)
        return (total, acoes, info.get("compliance"), viol0, violf,
                info.get("deadlock", False))

    peso = 5.0
    t0, a0, c0, _, _, dl0 = _rollout(0.0)
    tp, ap, cp, v0, vf, dlp = _rollout(peso)
    assert a0 == ap and c0 == cp, "dfam mudou a trajetoria (nao deveria)"
    assert not (dl0 or dlp), "rollout de teste caiu em deadlock; trocar semente"
    esperado = -peso * (vf - v0)
    assert abs((tp - t0) - esperado) < 1e-6, ((tp - t0), esperado)
    print(f"[OK] penalidade dfam: peso 0 inerte; retorno deslocado por "
          f"-peso*(dviol) = {esperado:+.4f} com trajetoria identica "
          f"(viol {v0:.3f} -> {vf:.3f})")


def test_v21_politica_fcn_smoke():
    """TopoFCNPolicy (06) constroi, treina alguns milhares de
    passos com MaskablePPO e salva/recarrega sem erro — a mesma politica
    usada por 08_treinar_ppo.py."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    import sys as _sys
    _sp = spec_from_file_location("politica_fcn",
                                  os.path.join(_DIR, "06_politica_fcn.py"))
    _pm = module_from_spec(_sp)
    _sys.modules["politica_fcn"] = _pm
    _sp.loader.exec_module(_pm)
    TopoFCNPolicy = _pm.TopoFCNPolicy
    fcn_policy_kwargs = _pm.fcn_policy_kwargs

    nx, ny = 12, 6

    def _init():
        env = TopoOptMaskEnv(nelx=nx, nely=ny, vol_frac=0.5,
                             problem_type="cantilever")
        return ActionMasker(env, lambda e: e.get_action_mask())

    vec = DummyVecEnv([_init, _init])
    t0 = time.time()
    model = MaskablePPO(TopoFCNPolicy, vec,
                        policy_kwargs=fcn_policy_kwargs(nx, ny),
                        n_steps=64, batch_size=64, n_epochs=2,
                        gamma=1.0, verbose=0, seed=7)
    model.learn(total_timesteps=2_000)

    ckpt = os.path.join(_DIR, "_smoke_fcn_ckpt")
    model.save(ckpt)
    recarregado = MaskablePPO.load(ckpt)
    env = TopoOptMaskEnv(nelx=nx, nely=ny, vol_frac=0.5, problem_type="cantilever")
    obs, _ = env.reset(seed=0)
    mask = env.get_action_mask()
    acao, _ = recarregado.predict(obs, action_masks=mask, deterministic=True)
    assert env.get_action_mask()[int(acao)] == 1.0, "acao prevista invalida pela mascara"
    os.remove(ckpt + ".zip")
    print(f"[OK] fumaca da politica FCN (2k passos, save/load, {time.time()-t0:.0f}s)")


def test_cache():
    env = TopoOptMaskEnv(nelx=20, nely=10, vol_frac=0.5)
    env.reset(seed=0)
    m1 = env.get_action_mask()
    m2 = env.get_action_mask()          # deve vir do cache
    assert m2 is m1, "cache nao reutilizado"
    a = int(np.nonzero(m1)[0][0])
    env.step(a)
    m3 = env.get_action_mask()
    assert m3 is not m1, "cache nao invalidado apos step"
    print("[OK] cache reutiliza por estado e invalida apos step")


def test_deadlock():
    env = TopoOptMaskEnv(nelx=20, nely=10, vol_frac=0.5)
    env.reset(seed=0)
    # estado artificial: barra horizontal 1xN (todo elemento interno e ponto
    # de articulacao) -> a mascara esvazia rapido apos remover as pontas
    env.density[:] = 1e-4
    env.density[env.nely // 2, :] = 1.0
    env._mask_dirty = True
    env.n_removed = 0
    mask = env.get_action_mask()
    assert mask.sum() <= 2, f"esperava <=2 acoes na barra, veio {int(mask.sum())}"

    for _ in range(env.n_elements):
        mask = env.get_action_mask()
        valid = np.nonzero(mask)[0]
        if len(valid) == 0:
            break
        obs, r, done, trunc, info = env.step(int(valid[0]))
        assert np.all(np.isfinite(obs)), "obs com NaN/inf"
        assert np.isfinite(r), "reward com NaN/inf"
        if done:
            assert info.get("deadlock", False) or env.n_removed >= env.n_to_remove, \
                "terminou sem deadlock nem VF alvo"
            if info.get("deadlock", False):
                print("[OK] deadlock termina o episodio de forma limpa "
                      f"(apos {env.n_removed} remocoes, reward finito)")
                return
    print("[OK] barra esgotada sem deadlock (mascara protegeu ate o fim)")


def test_smoke_treino():
    _t = spec_from_file_location("treinar", os.path.join(_DIR, "08_treinar_ppo.py"))
    _tm = module_from_spec(_t)
    _t.loader.exec_module(_tm)

    t0 = time.time()
    _tm.train(nelx=12, nely=6, vol_frac=0.5, total_timesteps=2_000,
              n_envs=1, seed=7, problem_type="cantilever",
              n_steps=64, batch_size=64, n_epochs=2,
              eval_freq=10_000_000)
    print(f"[OK] fumaca de treino MaskablePPO (2k passos, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    test_equivalencia()
    test_cache()
    test_deadlock()
    test_espessura_minima()
    test_simetria_horizontal()
    test_dfam()
    try:
        import torch  # noqa: F401
        import sb3_contrib  # noqa: F401
        _tem_sb3 = True
    except ImportError:
        _tem_sb3 = False
        print("[--] torch/sb3-contrib ausentes: fumacas de politica e treino puladas")
    if _tem_sb3:
        test_v21_politica_fcn_smoke()
        test_smoke_treino()
    print("\ntodos os testes passaram.")
