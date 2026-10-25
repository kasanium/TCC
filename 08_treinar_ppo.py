"""
treinador ppo unico dos dois dominios (--problem cantilever|automotive):
maskableppo + politica totalmente convolucional (06) sobre o ambiente 05.
recompensa telescopica com gamma=1.0, n_steps automatico cobrindo o
episodio inteiro com folga, batch 1024, 4 epocas, ent_coef 3e-3 (recalibrado
para ~900 acoes). callback salva best_model.zip pela menor compliancia num
episodio greedy deterministico. --resume restaura pesos com protecao contra
o _setup_model do sb3 (que recriaria a politica do zero em silencio) e
imprime o c_greedy do checkpoint carregado antes de gastar horas de treino.
--simetria horizontal remove pares elemento+espelho (usado no cantilever).
--dfam_peso ativa a penalidade de overhang do ambiente (ver 05); variacoes
de seed e de dfam salvam em diretorios com sufixo, preservando o canonico.

resultados de referencia deste tcc (treino do zero, seed 42):
    cantilever 20x10 vf=0.50 --simetria horizontal, 1.5m passos (C = 70.85)
    automotivo 30x30 vf=0.30, 3m passos (C = 20.32; sementes 7/13/21/42/99:
    20.33 +- 0.03)
"""

import os
import sys
import time
import argparse

import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    raise ImportError("sb3-contrib necessario: pip install sb3-contrib")

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("rl_env_mask", os.path.join(_DIR, "05_ambiente_rl.py"))
_env_mod = module_from_spec(_spec)
_spec.loader.exec_module(_env_mod)
TopoOptMaskEnv = _env_mod.TopoOptMaskEnv
topo_mask_fn = _env_mod.topo_mask_fn

_spec_pol = spec_from_file_location("politica_fcn",
                                    os.path.join(_DIR, "06_politica_fcn.py"))
_pol_mod = module_from_spec(_spec_pol)
sys.modules["politica_fcn"] = _pol_mod   # nome gravado nos checkpoints
_spec_pol.loader.exec_module(_pol_mod)
TopoFCNPolicy = _pol_mod.TopoFCNPolicy
fcn_policy_kwargs = _pol_mod.fcn_policy_kwargs

try:                       # o SB3 quebra se tensorboard_log apontar para um
    import tensorboard     # diretorio sem o tensorboard instalado
    _TB = True
except ImportError:
    _TB = False
    print("aviso: tensorboard nao instalado; logs de tensorboard desligados")


def _tb(log_dir):
    return log_dir if _TB else None


class MonitorCallback(BaseCallback):
    """Imprime recompensa, comprimento de episodio e KL ao longo do treino.

    Com a recompensa telescopica, ep_rew e interpretavel diretamente:
    ep_rew = reward_scale*log(c_0/c_T), menos a penalidade de deadlock. Com
    reward_scale=10 e c_0 = 7.68 no caso automotivo, C = 20.8 corresponde a
    ep_rew ~ -10.0 e C = 93.5 a ep_rew ~ -25.0. Quanto MAIOR (menos
    negativo), melhor.
    """

    def __init__(self, total_timesteps, print_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            fps = max(1, int(self.num_timesteps / elapsed))
            pct = self.num_timesteps / self.total_timesteps * 100
            eta = (self.total_timesteps - self.num_timesteps) / fps / 60
            buf = self.model.ep_info_buffer
            ep_rew = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
            ep_len = float(np.mean([e["l"] for e in buf])) if buf else float("nan")
            kl = self.logger.name_to_value.get("train/approx_kl", float("nan"))
            print(f"[{pct:5.1f}%] steps={self.num_timesteps:>8d} | "
                  f"ep_rew={ep_rew:>8.3f} | ep_len={ep_len:>5.0f} | "
                  f"kl={kl:.4f} | fps={fps} | eta={eta:.0f}min")
        return True


class MelhorPorCompliancia(BaseCallback):
    """Salva best_model.zip pelo criterio que interessa: menor compliancia num
    episodio greedy deterministico, avaliado periodicamente."""

    def __init__(self, nelx, nely, vol_frac, problem_type, mod_dir,
                 eval_freq=200_000, melhor_inicial=float("inf"),
                 simetria=None, verbose=0):
        super().__init__(verbose)
        self.nelx, self.nely = nelx, nely
        self.vol_frac, self.problem_type = vol_frac, problem_type
        self.mod_dir = mod_dir
        self.eval_freq = eval_freq
        self.simetria = simetria
        # ao retomar um checkpoint, parte-se da compliancia DELE: senao o
        # callback sobrescreve um best_model melhor com a primeira avaliacao
        # que fizer, so porque comecou em infinito
        self.melhor = float(melhor_inicial)
        self._prox = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self._prox:
            return True
        self._prox = self.num_timesteps + self.eval_freq

        env = TopoOptMaskEnv(nelx=self.nelx, nely=self.nely,
                             vol_frac=self.vol_frac,
                             problem_type=self.problem_type,
                             simetria=self.simetria)
        wenv = ActionMasker(env, topo_mask_fn)
        obs, _ = wenv.reset(seed=0)
        done, info = False, {}
        while not done:
            acao, _ = self.model.predict(obs, action_masks=wenv.action_masks(),
                                         deterministic=True)
            obs, _, term, trunc, info = wenv.step(int(acao))
            done = term or trunc
        c = float(info.get("compliance", float("inf")))
        dl = info.get("deadlock", False)
        marca = ""
        if not dl and c < self.melhor:
            self.melhor = c
            self.model.save(os.path.join(self.mod_dir, "best_model"))
            marca = "  <- novo melhor, salvo"
        print(f"   [aval @ {self.num_timesteps:>8d}] C_greedy = {c:.4f} | "
              f"{env.n_removed}/{env.n_to_remove} | deadlock={dl}{marca}")
        return True


def _avaliar_greedy(model, nelx, nely, vol_frac, problem_type, seed=0,
                    simetria=None):
    """Um episodio greedy deterministico. Usado tanto pelo callback de melhor
    modelo quanto pela verificacao de sanidade logo apos um --resume."""
    env = TopoOptMaskEnv(nelx=nelx, nely=nely, vol_frac=vol_frac,
                         problem_type=problem_type, simetria=simetria)
    wenv = ActionMasker(env, topo_mask_fn)
    obs, _ = wenv.reset(seed=seed)
    done, info = False, {}
    while not done:
        acao, _ = model.predict(obs, action_masks=wenv.action_masks(),
                                deterministic=True)
        obs, _, term, trunc, info = wenv.step(int(acao))
        done = term or trunc
    return float(info.get("compliance", float("inf")))


def make_env(nelx, nely, vol_frac, problem_type, thin_threshold,
             reward_mode, reward_scale, deadlock_penalty, simetria=None,
             dfam_peso=0.0, dfam_base="baixo"):
    def _init():
        env = TopoOptMaskEnv(nelx=nelx, nely=nely, vol_frac=vol_frac,
                             problem_type=problem_type,
                             thin_threshold=thin_threshold,
                             reward_mode=reward_mode,
                             reward_scale=reward_scale,
                             deadlock_penalty=deadlock_penalty,
                             simetria=simetria,
                             dfam_peso=dfam_peso, dfam_base=dfam_base)
        return ActionMasker(env, topo_mask_fn)
    return _init


def build_vec_env(nelx, nely, vol_frac, n_envs, log_dir,
                  problem_type="automotive", thin_threshold=0.0,
                  reward_mode="log_delta", reward_scale=10.0,
                  deadlock_penalty=10.0, simetria=None,
                  dfam_peso=0.0, dfam_base="baixo"):
    fns = [make_env(nelx, nely, vol_frac, problem_type, thin_threshold,
                    reward_mode, reward_scale, deadlock_penalty, simetria,
                    dfam_peso, dfam_base)
           for _ in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)
    return VecMonitor(vec, filename=os.path.join(log_dir, "treino"))


def new_model(env, log_dir, seed, nelx, nely, n_steps=768, gamma=1.0,
              batch_size=1024, n_epochs=4, ent_coef=3e-3, lr=3e-4,
              clip_range=0.2, target_kl=None):
    """MaskablePPO com a politica totalmente convolucional; ver itens 1 e 3 da
    docstring do modulo para o porque de cada desvio em relacao a v19."""
    return MaskablePPO(
        TopoFCNPolicy, env,
        policy_kwargs=fcn_policy_kwargs(nelx, nely),
        verbose=0,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        target_kl=target_kl,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=_tb(log_dir),
        device="auto",
        seed=seed,
    )


def load_model(resume, env, log_dir, lr=3e-4, ent_coef=None, n_steps=None,
               batch_size=None, n_epochs=None, gamma=None,
               clip_range=0.2, target_kl=None):
    """retoma um checkpoint sobrescrevendo os hiperparametros de rollout."""
    custom_objects = {
        "lr_schedule": lambda _: lr,
        "clip_range": lambda _: clip_range,
    }
    model = MaskablePPO.load(resume, env=env, custom_objects=custom_objects,
                             device="auto", tensorboard_log=_tb(log_dir))
    model.set_env(env)
    if ent_coef is not None:
        print(f"  ent_coef: {model.ent_coef} -> {ent_coef}")
        model.ent_coef = ent_coef
    if gamma is not None:
        model.gamma = gamma
    model.target_kl = target_kl
    if n_steps is not None and n_steps != model.n_steps:
        print(f"  n_steps: {model.n_steps} -> {n_steps}")
        # atencao: _setup_model() do sb3 recria self.policy do zero, o que
        # apagaria em silencio os pesos recem-carregados (o treino seguiria
        # a partir de pesos aleatorios). salva e restaura o state_dict em
        # volta da chamada; o otimizador e reiniciado de proposito.
        estado = {k: v.clone() for k, v in model.policy.state_dict().items()}
        model.n_steps = n_steps
        model._setup_model()
        model.policy.load_state_dict(estado)
    if batch_size is not None:
        model.batch_size = batch_size
    if n_epochs is not None:
        model.n_epochs = n_epochs
    return model


def train_phase(nelx, nely, vol_frac, total_timesteps, n_envs, seed,
                run_name, name_prefix, resume=None, n_steps=768, gamma=1.0,
                batch_size=1024, n_epochs=4, ent_coef=3e-3, lr=3e-4,
                clip_range=0.2, target_kl=None,
                thin_threshold=0.0, problem_type="automotive",
                reward_mode="log_delta", reward_scale=10.0,
                deadlock_penalty=10.0, eval_freq=200_000, simetria=None,
                dfam_peso=0.0, dfam_base="baixo"):
    log_dir = os.path.join(_DIR, "resultados", "logs", run_name)
    mod_dir = os.path.join(_DIR, "resultados", "dados", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)

    env = build_vec_env(nelx, nely, vol_frac, n_envs, log_dir,
                        problem_type=problem_type,
                        thin_threshold=thin_threshold,
                        reward_mode=reward_mode, reward_scale=reward_scale,
                        deadlock_penalty=deadlock_penalty, simetria=simetria,
                        dfam_peso=dfam_peso, dfam_base=dfam_base)
    if resume:
        print(f"  retomando de: {resume}")
        model = load_model(resume, env, log_dir, lr=lr, ent_coef=ent_coef,
                           n_steps=n_steps, batch_size=batch_size,
                           n_epochs=n_epochs, gamma=gamma,
                           clip_range=clip_range, target_kl=target_kl)
        # confere que os pesos sobreviveram ao carregamento antes de gastar
        # horas de treino em cima de um checkpoint que virou ruido
        c0 = _avaliar_greedy(model, nelx, nely, vol_frac, problem_type,
                             simetria=simetria)
        print(f"  checkpoint carregado: C_greedy = {c0:.4f}")
    else:
        model = new_model(env, log_dir, seed, nelx, nely, n_steps=n_steps,
                          gamma=gamma, batch_size=batch_size,
                          n_epochs=n_epochs, ent_coef=ent_coef, lr=lr,
                          clip_range=clip_range, target_kl=target_kl)

    melhor_inicial = c0 if resume else float("inf")
    ckpt_freq = max(total_timesteps // 10, 100_000)
    callbacks = [
        MonitorCallback(total_timesteps, print_freq=10_000),
        MelhorPorCompliancia(nelx, nely, vol_frac, problem_type, mod_dir,
                             eval_freq=eval_freq,
                             melhor_inicial=melhor_inicial,
                             simetria=simetria),
        CheckpointCallback(save_freq=max(1, ckpt_freq // n_envs),
                           save_path=mod_dir, name_prefix=name_prefix),
    ]
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = os.path.join(mod_dir, f"{name_prefix}_final")
    model.save(final_path)
    env.close()
    return final_path


# Referencias medidas no mesmo espaco de acoes do agente, uma execucao
# deterministica cada, semente 0:
#   (SIMP binarizado, melhor politica gulosa, raio de filtro desse professor)
# O raio otimo do professor NAO e o mesmo nos dois casos: em 30x30 o filtro
# rmin=1.5 melhora o guloso (22.78 -> 20.82 na geometria anterior a 2026-10; 20.42 com os furos simetricos), mas em 20x10 ele piora
# (71.66 -> 82.45), porque o nucleo borra o campo numa fracao grande dos 10
# elementos de altura do dominio e o argmin perde discriminacao.
REFERENCIAS = {
    ("automotive", 30, 30, 0.30): (20.72, 20.42, 1.5),
    ("cantilever", 20, 10, 0.50): (67.31, 71.66, 0.0),
}
# com simetria horizontal (v22): o par elemento+espelho e mais restritivo,
# o filtro inverte de papel (ajuda em vez de atrapalhar) — ver
# 05_ambiente_rl.py (rodar_guloso)
REFERENCIAS_SIMETRIA = {
    ("cantilever", 20, 10, 0.50): (67.31, 78.90, 1.5),
}


def train(nelx=30, nely=30, vol_frac=0.3, total_timesteps=3_000_000,
          n_envs=32, seed=42, resume=None, problem_type="automotive", **kw):
    tag = "automotivo" if problem_type == "automotive" else "cantilever"
    run_name = f"v21_{tag}_{nelx}x{nely}"
    # o diretorio canonico (usado por 09/11) e o do seed 42 sem dfam;
    # variacoes ganham sufixo para preservar o checkpoint de referencia
    if seed != 42:
        run_name += f"_seed{seed}"
    if kw.get("dfam_peso", 0.0) > 0.0:
        run_name += f"_dfam{kw['dfam_peso']:g}"
    print("=" * 70)
    print(f"  PPO v21 — HARD ACTION MASKING + POLITICA FCN — {tag.capitalize()}")
    print(f"  Malha: {nelx}x{nely} | VF: {vol_frac} | Steps: {total_timesteps:,}")
    print(f"  n_envs {n_envs} | n_steps {kw.get('n_steps')} | "
          f"gamma {kw.get('gamma')} | batch {kw.get('batch_size')} | "
          f"epochs {kw.get('n_epochs')} | ent_coef {kw.get('ent_coef')}")
    print(f"  recompensa: {kw.get('reward_mode')} "
          f"(escala {kw.get('reward_scale')})")
    if kw.get("dfam_peso", 0.0) > 0.0:
        print(f"  penalidade dfam: peso {kw['dfam_peso']:g} | "
              f"base {kw.get('dfam_base')}")
    # referencias medidas no MESMO espaco de acoes (ver 05_ambiente_rl.py (rodar_guloso))
    tabela_ref = (REFERENCIAS_SIMETRIA if kw.get("simetria") == "horizontal"
                 else REFERENCIAS)
    ref = tabela_ref.get((problem_type, nelx, nely, round(vol_frac, 2)))
    if ref:
        print(f"  referencias: SIMP binarizado C={ref[0]:.2f} | "
              f"melhor guloso C={ref[1]:.2f} (professor com filtro rmin={ref[2]})")
    print("=" * 70)
    t0 = time.time()
    final = train_phase(nelx, nely, vol_frac, total_timesteps, n_envs, seed,
                        run_name=run_name, name_prefix=f"ppo_v21_{tag}",
                        resume=resume, problem_type=problem_type, **kw)
    print(f"\nacabou em {(time.time()-t0)/3600:.1f}h | salvo: {final}.zip")
    print(f"melhor por compliancia: "
          f"{os.path.join(os.path.dirname(final), 'best_model.zip')}")
    return final


def train_curriculum(nelx=30, nely=30, vf_schedule=(0.60, 0.45, 0.30),
                     steps_per_phase=(1_000_000, 1_000_000, 4_000_000),
                     n_envs=32, seed=42, resume=None, **kw):
    """Curriculo de fracao de volume, mantido da v18. Cada fase retoma os
    pesos da anterior; o espaco de acoes nao muda entre fases."""
    assert len(vf_schedule) == len(steps_per_phase)
    tag = ("automotivo" if kw.get("problem_type", "automotive") == "automotive"
           else "cantilever")
    print("=" * 70)
    print("  PPO v21 — CURRICULO DE FRACAO DE VOLUME")
    print(f"  VF: {' -> '.join(f'{v:.2f}' for v in vf_schedule)}")
    print("=" * 70)
    t0 = time.time()
    final = None
    for i, (vf, steps) in enumerate(zip(vf_schedule, steps_per_phase)):
        print(f"\n--- Fase {i+1}/{len(vf_schedule)} | VF={vf:.2f} | "
              f"{steps:,} steps ---")
        run_name = f"v21_{tag}_{nelx}x{nely}_curr_vf{int(vf*100):02d}"
        final = train_phase(nelx, nely, vf, steps, n_envs, seed,
                            run_name=run_name,
                            name_prefix=f"ppo_curr_vf{int(vf*100):02d}",
                            resume=resume, **kw)
        resume = final + ".zip"

    canon_dir = os.path.join(_DIR, "resultados", "dados",
                             f"v21_{tag}_{nelx}x{nely}")
    os.makedirs(canon_dir, exist_ok=True)
    canon_path = os.path.join(canon_dir, f"ppo_v21_{tag}_final")
    MaskablePPO.load(final, device="auto").save(canon_path)
    print(f"\ncurriculo concluido em {(time.time()-t0)/3600:.1f}h")
    print(f"checkpoint final (canonico): {canon_path}.zip")
    return canon_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nelx", type=int, default=30)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--vol_frac", type=float, default=0.3)
    parser.add_argument("--problem", type=str, default="automotive",
                        choices=["automotive", "cantilever"])
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=None,
                        help="rollout por env; se omitido, e calculado para "
                             "cobrir o episodio inteiro com 25%% de folga "
                             "(automotivo 30x30 VF=0.30 -> 768; cantilever "
                             "20x10 VF=0.50 -> 128)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="1.0 com a recompensa telescopica: o retorno nao "
                             "descontado E o objetivo do problema")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--ent_coef", type=float, default=3e-3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="0.2 para treino do zero; 0.05 ao retomar um "
                             "checkpoint, para refinar sem dissolver a politica")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="aborta a epoca quando a divergencia de KL passa "
                             "desse valor; util ao retomar checkpoints")
    parser.add_argument("--reward_mode", type=str, default="log_delta",
                        choices=["log_delta", "brown"],
                        help="'brown' reproduz a recompensa da v19/v20")
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--deadlock_penalty", type=float, default=10.0)
    parser.add_argument("--thin_threshold", type=float, default=0.0,
                        help="restricao de espessura minima da v20; medida "
                             "como inerte na faixa calibrada, deixe em 0.0")
    parser.add_argument("--dfam_peso", type=float, default=0.0,
                        help="peso da penalidade de overhang de 45 graus; "
                             "0 desliga (padrao)")
    parser.add_argument("--dfam_base", type=str, default="baixo",
                        choices=["baixo", "cima", "esquerda", "direita"],
                        help="lado da base de impressao para a penalidade")
    parser.add_argument("--simetria", type=str, default=None,
                        choices=[None, "horizontal"],
                        help="cada acao remove o elemento e seu espelho "
                             "horizontal juntos; requer BCs simetricas "
                             "(v22, ver 05_ambiente_rl.py)")
    parser.add_argument("--eval_freq", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint .zip para retomar o treino")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--vf_schedule", type=float, nargs="+",
                        default=[0.60, 0.45, 0.30])
    parser.add_argument("--steps_per_phase", type=int, nargs="+",
                        default=[1_000_000, 1_000_000, 4_000_000])
    args = parser.parse_args()

    if args.n_steps is None:
        _probe = TopoOptMaskEnv(nelx=args.nelx, nely=args.nely,
                                vol_frac=args.vol_frac,
                                problem_type=args.problem,
                                simetria=args.simetria)
        # sob simetria cada acao remove 2 elementos (ver 05_ambiente_rl.py,
        # v22), entao o episodio tem ~metade dos passos
        ep_len = _probe.n_to_remove // 2 + 1 if args.simetria else _probe.n_to_remove
        args.n_steps = int(np.ceil(ep_len * 1.25 / 64) * 64)
        print(f"n_steps automatico: {args.n_steps} "
              f"(episodio de ~{ep_len} passos)")

    comum = dict(n_steps=args.n_steps, gamma=args.gamma,
                 batch_size=args.batch_size, n_epochs=args.n_epochs,
                 ent_coef=args.ent_coef, lr=args.lr,
                 clip_range=args.clip_range, target_kl=args.target_kl,
                 thin_threshold=args.thin_threshold,
                 problem_type=args.problem, reward_mode=args.reward_mode,
                 reward_scale=args.reward_scale,
                 deadlock_penalty=args.deadlock_penalty,
                 eval_freq=args.eval_freq, simetria=args.simetria,
                 dfam_peso=args.dfam_peso, dfam_base=args.dfam_base)

    if args.curriculum:
        train_curriculum(nelx=args.nelx, nely=args.nely,
                         vf_schedule=tuple(args.vf_schedule),
                         steps_per_phase=tuple(args.steps_per_phase),
                         n_envs=args.n_envs, seed=args.seed,
                         resume=args.resume, **comum)
    else:
        train(nelx=args.nelx, nely=args.nely, vol_frac=args.vol_frac,
              total_timesteps=args.steps, n_envs=args.n_envs,
              seed=args.seed, resume=args.resume, **comum)
