"""
ENAFYD 2024 — Domain-specific physical activity and social inequalities.
Consolidated, reproducible pipeline (Cells 2A–7 of the analysis notebook).

Usage:
    python analysis_pipeline.py --data data/BASE_DE_DATOS_ADULTOS.sav --out results/
"""
import argparse
import os
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib as mpl

MAP_FREQ = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0, 9: np.nan}
DOMINIOS = {"lab": "1", "esc": "2", "tlibre": "3", "dom": "4", "trans": "5"}
FORMULA_BASE = ("{y} ~ C(nse_cat) + C(sexo, Treatment('Hombre')) + edad_c "
                "+ C(area, Treatment('Urbana')) "
                "+ C(pueblo_orig, Treatment('No')) "
                "+ C(discapacidad, Treatment('No')) "
                "+ C(migrante, Treatment('No'))")


def construir_minutos(df, trunc=None, filtro_intensidad=False):
    """Weekly domain MVPA minutes under explicit, pre-specified rules."""
    out = pd.DataFrame(index=df.index)
    for d, num in DOMINIOS.items():
        dias = df[f"p26_{num}"].map(MAP_FREQ)
        mins_dia = df[f"p27_{num}"].copy()
        if trunc is not None:
            mins_dia = mins_dia.clip(upper=trunc)
        m = dias * mins_dia
        m[df[f"p26_{num}"] == 9] = 0
        m[dias == 0] = 0
        if filtro_intensidad:
            m[df[f"p28_{num}"] == 1] = 0
        out[f"min_{d}"] = m
    out["min_total"] = out.sum(axis=1)
    out["min_electiva"] = out["min_tlibre"] + out["min_trans"]
    out["cumple_oms"] = (out["min_total"] >= 150).astype(int)
    out["cumple_oms_elect"] = (out["min_electiva"] >= 150).astype(int)
    return out


def construir_equidad(df):
    df["sexo"] = df["p5"].map({1: "Mujer", 2: "Hombre"})
    df["edad"] = df["p4"]
    df["nse_cat"] = df["nse"].map({1: "Alto", 2: "Medio", 3: "Bajo"})
    df["area"] = df["p3"].map({1: "Urbana", 2: "Rural"})
    p13 = df["p13"].astype(str).str.strip()
    df["pueblo_orig"] = pd.Series(
        np.select([p13 == "10", p13.isin([str(i) for i in range(1, 10)])],
                  ["No", "Sí"], default=None), index=df.index, dtype="object")
    p11 = df["p11_5"].astype(str).str.strip()
    df["discapacidad"] = pd.Series(
        np.select([p11 == "5.1", p11 == "5.2"], ["No", "Sí"], default=None),
        index=df.index, dtype="object")
    p12 = df["p12"].astype(str).str.strip()
    df["migrante"] = pd.Series(
        np.select([p12 == "1", p12.isin([str(i) for i in range(2, 8)])],
                  ["No", "Sí"], default=None), index=df.index, dtype="object")
    return df


def poisson_pr(data, y):
    m = smf.glm(FORMULA_BASE.format(y=y), data=data,
                family=sm.families.Poisson(),
                freq_weights=data["pond"]).fit(cov_type="HC1")
    res = pd.DataFrame({"PR": np.exp(m.params),
                        "IC_low": np.exp(m.conf_int()[0]),
                        "IC_high": np.exp(m.conf_int()[1]),
                        "p": m.pvalues}).drop("Intercept")
    return m, res.round(3)


def ridit_ponderado(data, var, orden, pesos="pond"):
    w = data.groupby(var, observed=True)[pesos].sum().reindex(orden)
    prop = w / w.sum()
    acumulada = prop.cumsum() - prop / 2
    return data[var].map(acumulada.to_dict()).astype(float)


def rii_sii(data, y):
    ajuste = "C(sexo, Treatment('Hombre')) + edad_c"
    m_rii = smf.glm(f"{y} ~ ridit_nse + {ajuste}", data=data,
                    family=sm.families.Poisson(),
                    freq_weights=data["pond"]).fit(cov_type="HC1")
    m_sii = smf.wls(f"{y} ~ ridit_nse + {ajuste}", data=data,
                    weights=data["pond"]).fit(cov_type="HC1")
    return (np.exp(m_rii.params["ridit_nse"]),
            np.exp(m_rii.conf_int().loc["ridit_nse"]).values,
            m_sii.params["ridit_nse"] * 100,
            (m_sii.conf_int().loc["ridit_nse"] * 100).values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, meta = pyreadstat.read_sav(args.data)
    df = construir_equidad(df)

    # Scenario convergence audit (A–D) vs official index
    escenarios = {"A": (None, False), "B": (180, False),
                  "C": (180, True), "D": (None, True)}
    with open(os.path.join(args.out, "scenario_convergence.txt"), "w") as f:
        for k, (t, fi) in escenarios.items():
            esc = construir_minutos(df, t, fi)
            tab = pd.crosstab(df["ÍndiceGEN"], esc["cumple_oms"])
            act = 100 * tab.loc["Activo", 1] / tab.loc["Activo"].sum()
            ina = 100 * tab.loc["Inactivo", 1] / tab.loc["Inactivo"].sum()
            f.write(f"Scenario {k}: active->meets {act:.1f}% | "
                    f"inactive->meets {ina:.1f}%\n")

    # Primary: Scenario C
    final = construir_minutos(df, 180, True)
    for c in final.columns:
        df[c] = final[c]

    vars_modelo = ["cumple_oms", "cumple_oms_elect", "nse_cat", "sexo",
                   "edad", "area", "pueblo_orig", "discapacidad",
                   "migrante", "pond"]
    analitica = df[vars_modelo].dropna().copy()
    analitica["nse_cat"] = pd.Categorical(analitica["nse_cat"],
                                          categories=["Alto", "Medio", "Bajo"])
    analitica["edad_c"] = (analitica["edad"] - analitica["edad"].mean()) / 10

    # Models 1 & 2
    m1, r1 = poisson_pr(analitica, "cumple_oms")
    m2, r2 = poisson_pr(analitica, "cumple_oms_elect")
    r1.to_csv(os.path.join(args.out, "model1_total_PR.csv"))
    r2.to_csv(os.path.join(args.out, "model2_elective_PR.csv"))

    # RII / SII
    analitica["ridit_nse"] = ridit_ponderado(analitica, "nse_cat",
                                             ["Bajo", "Medio", "Alto"])
    with open(os.path.join(args.out, "rii_sii.txt"), "w") as f:
        for y, tag in [("cumple_oms", "total"), ("cumple_oms_elect", "elective")]:
            rii, rii_ic, sii, sii_ic = rii_sii(analitica, y)
            f.write(f"{tag}: RII={rii:.2f} ({rii_ic[0]:.2f}-{rii_ic[1]:.2f}) | "
                    f"SII={sii:+.1f}pp ({sii_ic[0]:+.1f} to {sii_ic[1]:+.1f})\n")

    print("Pipeline completed. Outputs in", args.out)


if __name__ == "__main__":
    main()
