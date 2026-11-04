"""escolha das metas de volume do benchmark SO com o mef, antes de rodar a u-net.

roda o dqn original de brown (sem substituto) na ponte e no console com
vol em {0.35, 0.40, 0.45} e grava cada tentativa. criterio: o original
"completa a etapa 24x24", isto e, vf final a menos de 0.01 da meta e mais de
20 remocoes na malha 24x24. entre as metas que passam, fica a mais proxima da
meta original do benchmark (0.40 na ponte, 0.50 no console). se nenhuma
passar, mantem a original e marca "original nao atinge a meta".
o cantilever fica em 0.35.
saida: resultados/metas_escolha.csv (todas as tentativas e a escolha)
"""
import os
import sys
import csv
from importlib.util import spec_from_file_location, module_from_spec

sys.argv = sys.argv[:1]
_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = spec_from_file_location("t03", os.path.join(_DIR, "t03_benchmark_brown_unet.py"))
t03 = module_from_spec(_spec)
_spec.loader.exec_module(t03)

METAS = [0.35, 0.40, 0.45]
ORIGINAL = {"2_Ponte": 0.40, "3_Console": 0.50}
TOL_VF = 0.01
MIN_REMOCOES = 20

if __name__ == "__main__":
    print(">> aquecimento (nao medido)")
    t03.run_brown(dict(t03.TEST_CASES["3_Console"]), use_surrogate=False)

    linhas, escolha = [], {}
    for caso, vol_orig in ORIGINAL.items():
        passam = []
        for vol in METAS:
            conf = dict(t03.TEST_CASES[caso])
            conf["vol"] = vol
            print(f"\n>> {caso} | meta {vol:.2f} | so mef")
            _, _, m = t03.run_brown(conf, use_surrogate=False)
            ok = abs(m["vf_final"] - vol) < TOL_VF and m["remocoes_24"] > MIN_REMOCOES
            print(f"   VF final {m['vf_final']:.3f} | remocoes 24x24 {m['remocoes_24']} | "
                  f"energia {m['se_real']:.4g} | tempo {m['tempo_s']:.1f}s | passa {ok}")
            linhas.append({"caso": caso, "meta": vol, "vf_final": round(m["vf_final"], 4),
                           "remocoes_24": m["remocoes_24"], "blocos2x2": round(m["blocos2x2"], 3),
                           "se_real": round(m["se_real"], 3), "tempo_s": round(m["tempo_s"], 2),
                           "passa": ok, "escolhida": False, "obs": ""})
            if ok:
                passam.append(vol)
        if passam:
            escolha[caso] = min(passam, key=lambda v: (abs(v - vol_orig), -v))
            obs = ""
        else:
            escolha[caso] = vol_orig
            obs = "original nao atinge a meta"
        for l in linhas:
            if l["caso"] == caso and l["meta"] == escolha[caso]:
                l["escolhida"] = True
                l["obs"] = obs
        if not passam:
            linhas.append({"caso": caso, "meta": vol_orig, "vf_final": "", "remocoes_24": "",
                           "blocos2x2": "", "se_real": "", "tempo_s": "", "passa": False,
                           "escolhida": True, "obs": obs + " (meta original mantida)"})

    out = os.path.join(_DIR, "resultados", "metas_escolha.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nescolha: cantilever 0.35 | " +
          " | ".join(f"{c} {v:.2f}" for c, v in escolha.items()))
    print(f"salvo: {out}")
