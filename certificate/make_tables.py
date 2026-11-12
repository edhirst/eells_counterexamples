"""LaTeX tables of the paper's appendix from the certificate outputs and the data files.

* tableA1.tex          Table A.1 (master-table): the certified constants for every Hopf invariant q,
                       one column per q (certificate/output/q{q}_D{D}_l{l}.json);
* coefficients.tex     the leading coefficients of the approximate solutions Wbar_q (Table B.1, tab:coefficients):
                       every basis field whose coefficient has modulus >= --threshold for some q, with
                       the field labelled by (type, n, j) so that rows are comparable across q;
* index_fields.tex     the test fields f_1, ..., f_5 of Proposition 3.2 (Table C.1, tab:index-fields):
                       their Gaussian-integer coefficients and the certified Rayleigh-quotient bounds.

Every number of tableA1.tex is an outward-rounded rigorous bound (rounded in the certificate); the
coefficients in coefficients.tex are rounded to --digits decimals for display only (the exact
IEEE doubles are in data/Wbar_q*_D*.json); the integers in index_fields.tex are exact.

Usage:  python -m certificate.make_tables certificate/output/q2_D40_l40.json ... certificate/output/q6_D184_l80.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from eells.fields import load_coefficients
from eells.reference import sobolev_constants
from eells.sectors import load_index_basis
from .rounding import round_down, round_up, sci

# rows of Table A.1: (certificate name, LaTeX symbol, significant digits for the display or None = as stored)
ROWS = [
    ("deg Wbar", r"$\deg\bar W$", None),
    ("l", r"$l$", None),
    ("C_residual", r"$\Cr{const:residual}=\varepsilon$", None),
    ("C_energy-lower", r"$\Cr{const:energy-lower}$", 13),
    ("C_energy-upper", r"$\Cr{const:energy-upper}$", 13),
    ("C_homotopy", r"$\Cr{const:homotopy}$", None),
    ("C_nl", r"$\Cr{const:non-linear}$", None),
    ("C_energy-lower2", r"$\Cr{const:energy-lower2}$", 13),
    ("C_energy-upper2", r"$\Cr{const:energy-upper2}$", 13),
    ("C_energyparameter", r"$\Cr{const:energyparameter}$", None),
    ("norm_X(Wbar)", r"$\|\bar W\|_X$", None),
    ("sup |Wbar|", r"$\|\bar W\|_{C^0}$", None),
    ("eta", r"$\eta$", None),
    ("b_1", r"$b_1$", None),
    ("b_0", r"$b_0$", None),
    ("gamma", r"$\gamma$", None),
    ("||A^-1||", r"$\|A^{-1}\|$", None),
    ("||B||", r"$\|B\|$", None),
    ("||C||", r"$\|C\|$", None),
    ("||A^-1 B||", r"$\|A^{-1}B\|$", None),
    ("||L^-1||_L2", r"$\|\mathscr L^{-1}\|_{L^2\to L^2}$", None),
    ("M", r"$\|\mathscr L^{-1}\|_{Y\to X}$", None),
    ("r", r"$r$", None),
    ("Mk", r"$Mk$", None),
    ("M(eps+kr)", r"$M(\varepsilon+kr)$", None),
    ("w_C0", r"$\|w\|_{C^0}$", None),
    ("delta", r"$\delta$", None),
    ("lambda*", r"$\lambda^*$", None),
]


def _value(row, digits):
    v, kind = row["value"], row["kind"]
    if digits is not None and kind in ("upper", "lower"):
        v = (round_up if kind == "upper" else round_down)(float(v), digits)
    rel = {"upper": r"\le ", "lower": r"\ge ", "exact": ""}[kind]
    return f"${rel}{sci(v)}$"


def master_table(certs):
    qs = [c["inputs"]["hopf_invariant"] for c in certs]
    rows = {c["inputs"]["hopf_invariant"]: {r["name"]: r for r in c["table_A1"]} for c in certs}
    first = rows[qs[0]]
    n = len(qs)
    lines = ["Hopf invariant $q$ & " + " & ".join(f"${q}$" for q in qs) + r"\\", r"\midrule"]
    for name, sym, digits in ROWS:
        cells = [(_value(rows[q][name], digits) if name in rows[q] else "--") for q in qs]
        if name == "M":                      # the bound M of the injectivity estimate, as used in Theorem 2.22 (thm:NK)
            cells = [c.replace(r"\le ", r"\le M=") for c in cells]
        lines.append(f"{sym} & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    # the exact constants and the interval enclosures used by the certificate (eells.reference.sobolev_constants),
    # displayed rounded outward to 16 decimals
    balls = dict(zip(("c_inf", "c_4", "c_5", "beta"), sobolev_constants(True)))
    for key, sym in (("c_inf", r"c_\infty"), ("c_4", "c_4"), ("c_5", "c_5"), ("beta", r"\beta")):
        lo = Decimal(float(balls[key].lower())).quantize(Decimal("1e-16"), rounding=ROUND_FLOOR)
        hi = Decimal(float(balls[key].upper())).quantize(Decimal("1e-16"), rounding=ROUND_CEILING)
        lines.append(f"${sym}$ & \\multicolumn{{{n}}}{{c}}{{${first[key]['value']}\\in[{lo},\\,{hi}]$}}\\\\[2pt]")
    body = "\n".join(lines)
    passed = all(c["passed"] for c in certs)
    caption = (r"Constants of the certificate for the approximate solutions $\bar W=\bar W_q$ of Hopf invariant "
               r"$q=2,\dots,6$ in \cref{thm:main}, with the cutoff $l$ of "
               r"\S\ref{subsection:linear-estimate}; $\Cr{const:homotopy}$ refers to the reference map "
               r"$V_q=(w^2-\frac14\bar z^{2q},\Re(z^qw))$. Upper bounds are "
               r"rounded up and lower bounds down. In every column $Mk<1$, $M(\varepsilon+kr)\le r$, "
               r"$\Cr{const:homotopy}-c_\infty r>0$ and $\|w\|_{C^0}<1$; for $q=2$, moreover, $\lambda^*+\delta<0$ "
               r"(\cref{prop:indexbound,prop:quadraticestimate}). The constants $c_\infty,c_4,c_5,\beta$ are the exact "
               r"values of \cref{prop:sobolev,prop:sobolevderivatives}; the certificates use interval enclosures of them, "
               r"which lie in the intervals shown.")
    assert passed, "a certificate in the list did not pass"
    return ("% generated by certificate/make_tables.py from " + ", ".join(c["_path"] for c in certs) + "\n"
            "\\begin{table}[H]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{l{'c' * n}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{master-table}}\n\\end{{table}}\n")


def coefficient_table(files, threshold=1e-2, digits=6):
    data = {}
    meta = {}
    for f in files:
        modes, values, d = load_coefficients(f)
        q = modes[0].q
        data[q] = {(m.kind, m.n, m.j): v for m, v in zip(modes, values)}
        meta[q] = (d["degree"], len(modes), f)
    qs = sorted(data)
    keys = set().union(*[set(d) for d in data.values()])
    keys = [k for k in keys if max(abs(data[q].get(k, 0.0)) for q in qs) >= threshold]
    keys.sort(key=lambda k: (k[0], abs(k[1]), -k[1], k[2]))
    lines = [r"type & $n$ & $j$ & " + " & ".join(f"$q={q}$" for q in qs) + r"\\", r"\midrule"]
    prev = None
    for k in keys:
        if prev is not None and (k[0], k[1]) != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = (k[0], k[1])
        cells = []
        for q in qs:
            v = data[q].get(k)
            cells.append("--" if v is None else f"${v:+.{digits}f}$".replace("+", "\\phantom{-}"))
        lines.append(f"{k[0]} & ${k[1]}$ & ${k[2]}$ & " + " & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{$\deg\bar W_q$} & " + " & ".join(f"${meta[q][0]}$" for q in qs) + r"\\")
    lines.append(r"\multicolumn{3}{l}{number of coefficients} & " + " & ".join(f"${meta[q][1]}$" for q in qs) + r"\\")
    body = "\n".join(lines)
    caption = (r"Leading coefficients $a_k$ of the approximate solutions $\bar W_q=\sum_k a_k\Psi_k$ in the orthonormal "
               r"basis of \cref{prop:harmonic-bases}: type U is the field "
               r"$(\Re Y_{m,n,j},\Im Y_{m,n,j},0)$ with $m=qn-2q$, type V is $\sqrt2(0,0,\Re Y_{m,n,j})$ with $m=qn$. "
               f"Listed are all basis fields whose coefficient has modulus at least ${sci(str(threshold))}$ for some $q$, "
               f"rounded to {digits} decimals. The complete lists, with the exact IEEE double coefficients used by "
               r"the certificate, are the files \texttt{data/Wbar\_q$q$\_D$D$.json} of the repository.")
    return ("% generated by certificate/make_tables.py from " + ", ".join(meta[q][2] for q in qs) + "\n"
            "\\begin{table}[H]\n\\centering\\small\n"
            f"\\begin{{tabular}}{{crr{'r' * len(qs)}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{tab:coefficients}}\n\\end{{table}}\n")


def _gauss(c):
    a, b = int(round(c.real)), int(round(c.imag))
    assert a == c.real and b == c.imag, "index test fields must have Gaussian-integer coefficients"
    if b == 0:
        return f"${a}$"
    if a == 0:
        return f"${b}i$" if abs(b) != 1 else ("$i$" if b > 0 else "$-i$")
    return f"${a}{'+' if b > 0 else '-'}{abs(b)}i$"


def index_table(basis_path, cert):
    fields, _ = load_index_basis(basis_path)
    ix = cert["details"]["index"]
    quot = {f["name"]: f["H1_rayleigh_upper"] for f in ix["fields"]}
    cols = []
    for f in fields:
        entries = [(f"${md.kind}_{{{md.m},{md.n},{md.j}}}$", _gauss(c)) for md, c in zip(f["modes"], f["coeffs"])]
        cols.append(entries)
    nrow = max(len(c) for c in cols)
    k = len(fields)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$f_{{{i + 1}}}\\in S_{{{f['nu']},{'+' if f['parity'] > 0 else '-'}}}$}}"
                      for i, f in enumerate(fields))
    lines = [head + r"\\", r"\midrule"]
    for r in range(nrow):
        cells = []
        for c in cols:
            cells += list(c[r]) if r < len(c) else ["", ""]
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\midrule")
    lines.append(" & ".join(f"\\multicolumn{{2}}{{c}}{{$\\le {round_up(quot[f['name']], 4)}$}}" for f in fields) + r"\\")
    body = "\n".join(lines)
    caption = (r"The test fields of \cref{prop:indexbound} ($q=2$): $f_a=\sum c\,\Phi$ with Gaussian-integer "
               r"coefficients $c$ and modes $\Phi\in\{P_{m,n,j},M_{m,n,j},T_{m,n,j}\}$ of \eqref{eq:basisPMT} in the "
               r"sector $S_{\nu,p}$ of $f_a$. They are rounded eigenvectors of the Galerkin matrix of $Q_{\bar W}$ on the "
               r"modes of degree $\le " + str(ix["max_degree"]) + r"$ of each sector. The last row is the certified bound "
               r"on $Q_{\bar W}(\xi)/\|\xi\|_{H^1}^2$ for $\xi=f_1$, respectively $\xi=\Re f_a,\Im f_a$ ($a\ge2$); "
               r"its maximum is $\lambda^*$.")
    spec = "lr" + r"@{\hspace{12pt}}lr" * (k - 1)
    return ("% generated by certificate/make_tables.py from " + str(basis_path) + " (exact integers)\n"
            "\\begin{table}[H]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{tab:index-fields}}\n\\end{{table}}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("certificates", nargs="+", help="certificate JSON files (one per q)")
    ap.add_argument("--index-basis", default="data/index_basis_q2.json")
    ap.add_argument("--threshold", type=float, default=1e-2)
    ap.add_argument("--digits", type=int, default=6)
    ap.add_argument("--out-dir", default="certificate/output")
    args = ap.parse_args()
    certs = []
    for p in args.certificates:
        c = json.loads(Path(p).read_text())
        c["_path"] = p
        certs.append(c)
    certs.sort(key=lambda c: c["inputs"]["hopf_invariant"])
    out = Path(args.out_dir)
    (out / "tableA1.tex").write_text(master_table(certs))
    (out / "coefficients.tex").write_text(coefficient_table([c["inputs"]["coefficients"] for c in certs],
                                                            args.threshold, args.digits))
    q2 = [c for c in certs if c["inputs"]["hopf_invariant"] == 2]
    written = ["tableA1.tex", "coefficients.tex"]
    if q2 and q2[0]["details"].get("index"):
        (out / "index_fields.tex").write_text(index_table(args.index_basis, q2[0]))
        written.append("index_fields.tex")
    print("wrote " + ", ".join(str(out / w) for w in written))


if __name__ == "__main__":
    main()
