import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from equations import flops, memory, bytes_moved, latency, energy
from equations import _walk, _layer_flops, _layer_bytes
import torch.nn as nn

RESULTS = Path("results")
K = 17
S_BASE = [32, 64, 128, 224, 256, 384, 512]
B_BASE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
S_CURVE = np.arange(32, 513, 16)
B_CURVE = np.unique(np.round(np.logspace(0, 8, 40, base=2)).astype(int))


def fit_theta(ok, V_m, P_idle, P_peak):
    fit = ok[~ok.is_validation]
    y = fit.latency_median_s.values

    launch_corner = ok[ok.B * ok.S ** 2 <= 2e4]
    t_0 = float(launch_corner.latency_median_s.min()) / K

    res = minimize_scalar(
        lambda lg: np.mean((np.log(latency(fit.S.values, fit.B.values, (t_0, 10 ** lg, V_m)))
                            - np.log(y)) ** 2),
        bounds=(10, np.log10(P_peak)), method="bounded")
    theta = (t_0, 10 ** res.x, V_m)

    F = flops(fit.S.values, fit.B.values)
    lat = latency(fit.S.values, fit.B.values, theta)
    e_flop = max(0.0, float(F @ (fit.energy_j.values - P_idle * lat) / (F @ F)))
    return theta, (P_idle, e_flop, 0.0, theta), len(launch_corner)


def plot_quantity(name, ok, PRED):
    pred, col, unit = PRED[name]
    fig = plt.figure(figsize=(16, 4.8))

    ax = fig.add_subplot(1, 3, 1)
    for S in S_BASE:
        sub = ok[ok.S == S].sort_values("B")
        if sub.empty:
            continue
        ln, = ax.plot(sub.B, sub[col], "o", ms=5, label=f"S={S} px")
        ax.plot(B_CURVE, pred(S, B_CURVE), "-", lw=1.2, color=ln.get_color(), alpha=.75)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("размер батча B [образцы]"); ax.set_ylabel(f"{name} [{unit}]")
    ax.set_title(f"{name}: срезы по B"); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(1, 3, 2)
    for B in B_BASE:
        sub = ok[ok.B == B].sort_values("S")
        if sub.empty:
            continue
        ln, = ax.plot(sub.S, sub[col], "o", ms=5, label=f"B={B}")
        ax.plot(S_CURVE, pred(S_CURVE, B), "-", lw=1.2, color=ln.get_color(), alpha=.75)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(S_BASE); ax.set_xticklabels(S_BASE, fontsize=8); ax.minorticks_off()
    ax.set_xlabel("сторона изображения S [px]"); ax.set_ylabel(f"{name} [{unit}]")
    ax.set_title(f"{name}: срезы по S"); ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    SS, BB = np.meshgrid(S_CURVE, B_CURVE)
    ax.plot_surface(SS, np.log2(BB), np.log10(pred(SS, BB)), cmap="viridis",
                    alpha=.55, linewidth=0, rstride=2, cstride=2)
    ax.scatter(ok.S, np.log2(ok.B), np.log10(ok[col]), c="crimson", s=14,
               depthshade=False, label="замер")
    ax.set_xlabel("S [px]"); ax.set_ylabel("log2 B")
    ax.set_zlabel(f"log10 {name} [{unit}]")
    ax.set_title(f"{name}: поверхность модели"); ax.legend(fontsize=7)

    fig.suptitle(f"{name}: точки — измерено, линии и поверхность — модель", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / f"{name.lower()}_vs_grid.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def plot_parity(ok, PRED, names):
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6))
    for ax, name in zip(axes, names):
        pred, col, unit = PRED[name]
        for sub, mk, lbl in [(ok[~ok.is_validation], "o", "калибровка"),
                             (ok[ok.is_validation], "^", "валидация")]:
            ax.loglog(sub[col], pred(sub.S.values, sub.B.values), mk, ms=6,
                      alpha=.75, label=lbl)
        lim = [ok[col].min() * .7, ok[col].max() * 1.4]
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        for k in (1.2, 1 / 1.2):
            ax.plot(lim, [v * k for v in lim], "k:", lw=.6)
        ax.set_xlabel(f"измерено, {name} [{unit}]")
        ax.set_ylabel(f"предсказано, {name} [{unit}]")
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Предсказание против замера (пунктир — коридор ±20%)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_map(ok, PRED, names):
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6))
    for ax, name in zip(axes, names):
        pred, col, unit = PRED[name]
        e = ok.assign(err=100 * (pred(ok.S.values, ok.B.values) - ok[col]) / ok[col])
        piv = e.pivot_table(index="B", columns="S", values="err")
        m = np.nanmax(np.abs(piv.values))
        im = ax.pcolormesh(np.arange(len(piv.columns) + 1),
                           np.arange(len(piv.index) + 1), piv.values,
                           cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set_xticks(np.arange(len(piv.columns)) + .5)
        ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(piv.index)) + .5)
        ax.set_yticklabels(piv.index, fontsize=6)
        ax.set_xlabel("S [px]"); ax.set_ylabel("B [образцы]")
        ax.set_title(f"{name}: (модель − замер)/замер")
        plt.colorbar(im, ax=ax, label="%")
    fig.suptitle("Относительная ошибка по сетке (S, B)", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oom(df, total_memory, gpu):
    fig, ax = plt.subplots(figsize=(7, 5))
    for oom_flag, mk, lbl in [(False, "o", "прошло"), (True, "x", "OOM")]:
        sub = df[df.oom == oom_flag]
        if len(sub):
            ax.scatter(sub.S, sub.B, marker=mk, s=45, label=lbl)
    bound = [max([b for b in range(1, 257) if memory(s, b) < total_memory], default=np.nan)
             for s in S_CURVE]
    ax.plot(S_CURVE, bound, "r-", lw=1.5, label="граница по memory()")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("сторона изображения S [px]")
    ax.set_ylabel("размер батча B [образцы]")
    ax.set_title(f"Граница OOM: {gpu}, {total_memory/2**30:.1f} ГиБ")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "oom_boundary.png", dpi=150)
    plt.close(fig)


def plot_regimes(ok, theta, P_peak):
    """Режимы на одной панели: достигнутая производительность от нагрузки."""
    t_0, V_c, V_m = theta
    F = flops(ok.S.values, ok.B.values)
    M = bytes_moved(ok.S.values, ok.B.values)
    ach = F / ok.latency_median_s.values
    work = (ok.B * ok.S ** 2).values
    w_split = V_c * K * t_0 / 17751

    n_launch = n_mem = n_comp = 0
    for S, B in zip(ok.S, ok.B):
        tc = tm = 0.0
        for layer, i_s, o_s in _walk(S, B):
            if isinstance(layer, nn.Flatten):
                continue
            fc = _layer_flops(layer, i_s, o_s) / V_c
            fm = _layer_bytes(layer, i_s, o_s) / V_m
            if fc >= fm:
                tc += fc
            else:
                tm += fm
        n_launch += K * t_0 >= max(tm, tc)
        n_mem += (tm > K * t_0) and (tm > tc)
        n_comp += (tc > K * t_0) and (tc >= tm)
    n_memory_bound = n_launch + n_mem

    fig, ax = plt.subplots(figsize=(9, 5.6))
    lo, hi = work.min() / 2, work.max() * 2
    ax.axvspan(lo, w_split, color="tab:blue", alpha=.07)
    ax.axvspan(w_split, hi, color="tab:green", alpha=.07)
    ax.axvline(w_split, ls=":", c="k", lw=1.2)

    ww = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(ww, 17751 * ww / (K * t_0), "b--", lw=1.3,
            label=f"memory-bound: время = K·t_0 = {K*t_0*1e6:.0f} мкс")
    ax.axhline(V_c, c="k", lw=1.6, label=f"compute-bound: V_c = {V_c/1e12:.2f} ТФЛОП/с")
    sc = ax.scatter(work, ach, c=np.log2(ok.B), cmap="viridis", s=28, zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(ach.min() / 3, V_c * 2.5)
    ax.text(.14, .07, "memory-bound\nвремя не зависит\nот нагрузки",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="tab:blue", weight="bold")
    ax.text(.72, .07, "compute-bound\nплато на V_c",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="tab:green", weight="bold")
    ax.text(w_split * .88, ach.min() / 3 * 1.6, f"граница B·S² = {w_split:.0f}",
            rotation=90, fontsize=8, va="bottom", ha="right")
    ax.text(.985, .975,
            f"по модели {n_memory_bound} точек memory-bound, {n_comp} compute-bound\n"
            f"в memory-bound области время определяют {K} запусков ядер,\n"
            f"а не трафик DRAM: он не доминирует ни в одной конфигурации",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=.9))
    ax.set_xlabel("нагрузка B·S² [образцы·px²]")
    ax.set_ylabel("достигнутая производительность F/t [FLOP/с]")
    ax.set_title("Режимы: точки — замер, линии — границы модели")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="center left")
    plt.colorbar(sc, ax=ax, label="log2 B")

    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "regimes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    old = json.loads((RESULTS / "theta.json").read_text()) if (RESULTS / "theta.json").exists() else {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-m", type=float, default=old.get("theta_latency", {}).get("V_m_bytes"))
    ap.add_argument("--p-idle", type=float, default=old.get("theta_energy", {}).get("P_idle_w"))
    ap.add_argument("--p-peak", type=float, default=old.get("P_peak", 8.1e12))
    ap.add_argument("--bw-peak", type=float, default=old.get("BW_peak", 320e9))
    args = ap.parse_args()
    if args.v_m is None or args.p_idle is None:
        ap.error("нужны --v-m и --p-idle (или существующий results/theta.json)")

    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "measurements.csv")
    ok = df[~df.oom].copy()
    fit, val = ok[~ok.is_validation], ok[ok.is_validation]

    theta, theta_energy, n_corner = fit_theta(ok, args.v_m, args.p_idle, args.p_peak)
    t_0, V_c, V_m = theta

    print(f"конфигураций {len(df)}, OOM {df.oom.sum()}, калибровка {len(fit)}, валидация {len(val)}")
    print(f"t_0 = {t_0*1e6:7.2f} мкс/ядро   (минимум по {n_corner} точкам launch-угла)")
    print(f"V_m = {V_m/1e9:7.1f} ГБ/с       (измерено; {V_m/args.bw_peak:5.1%} от паспорта)")
    print(f"V_c = {V_c/1e12:7.2f} TFLOP/с    (подогнано; {V_c/args.p_peak:5.1%} от паспорта)")
    if V_c > args.p_peak * 0.999:
        print("!! V_c упёрся в паспортный пик -> модель или замеры не в порядке")
    print(f"P_idle = {theta_energy[0]:.1f} Вт (измерено), "
          f"e_flop = {theta_energy[1]*1e12:.3f} пДж/FLOP")

    json.dump({"model": "per-layer roofline: K*t_0 + sum_l max(F_l/V_c, M_l/V_m)",
               "theta_latency": {"t_0_s": t_0, "V_c_flops": V_c, "V_m_bytes": V_m},
               "theta_energy": {"P_idle_w": theta_energy[0], "e_flop_j": theta_energy[1],
                                "e_byte_j": 0.0},
               "n_kernels": K, "P_peak": args.p_peak, "BW_peak": args.bw_peak,
               "epsilon_compute": V_c / args.p_peak,
               "epsilon_memory": V_m / args.bw_peak,
               "env": old.get("env", {})},
              open(RESULTS / "theta.json", "w"), indent=2)

    PRED = {
        "Memory":  (lambda S, B: memory(S, B),              "peak_allocated",   "байт"),
        "Latency": (lambda S, B: latency(S, B, theta),       "latency_median_s", "с"),
        "Energy":  (lambda S, B: energy(S, B, theta_energy), "energy_j",         "Дж"),
    }
    names = ("Memory", "Latency", "Energy")

    def mape(p, m):
        return 100 * np.mean(np.abs((p - m) / m))

    print(f"\n{'величина':<10} {'единицы':>8} {'fit MAPE':>10} {'val MAPE':>10} {'val max':>10}")
    for name in names:
        pred, col, unit = PRED[name]
        pf, pv = pred(fit.S.values, fit.B.values), pred(val.S.values, val.B.values)
        mf, mv = fit[col].values, val[col].values
        print(f"{name:<10} {unit:>8} {mape(pf, mf):9.2f}% {mape(pv, mv):9.2f}% "
              f"{100*np.max(np.abs((pv-mv)/mv)):9.1f}%")

    for name in names:
        plot_quantity(name, ok, PRED)
    plot_parity(ok, PRED, names)
    plot_error_map(ok, PRED, names)
    plot_oom(df, old.get("env", {}).get("total_memory", 15637086208),
             old.get("env", {}).get("gpu", "GPU"))
    plot_regimes(ok, theta, args.p_peak)
    print()
    for f in sorted((RESULTS / "figures").glob("*.png")):
        print(f"{f.name:28} {f.stat().st_size/1024:7.1f} КиБ")


if __name__ == "__main__":
    main()
