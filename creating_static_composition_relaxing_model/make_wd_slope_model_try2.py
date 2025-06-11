#!/usr/bin/env python3
import os
import re
import numpy as np
from os import system
from os.path import isfile

# -----------------------------------------------------------------------------
# 1) Build a “(n+2)×10” composition file with a linear ramp (n=1000),
#    ensuring no two q‐values are effectively identical. If raw_q_start ≤ eps_min,
#    we force the first ramp‐point at q_end/n so that q never stays exactly at zero.
#
#    Column order (10 species):
#       [ H1, He3, He4, C12, N14, O16, Ne20, Ne22, Mg24, Z_other ]
# -----------------------------------------------------------------------------
def conv_wd_slope(
    X_env,    # length‐10 list: envelope mass fractions (must sum to 1.0)
    X_core,   # length‐10 list: core mass fractions     (must sum to 1.0)
    m_core,   # fraction of total mass that is “core” 
    m_grad    # width of the linear compositional ramp in mass fraction 
):
    """
    Returns a string for a composition file with:
      • Header: “(n+2) 10”, where n=1000 steps in the ramp.
      • Row 1: q=0.000000          → envelope composition (X_env)
      • Rows 2..(n+1): n=1000 rows, linearly interpolating from X_env→X_core,
          between q_start = max(1−m_core−m_grad, eps_min>0) and q_end = 1−m_core.
      • Row (n+2): q=1.000000      → core composition (X_core)

    If raw_q_start = 1−m_core−m_grad ≤ eps_min, we set q_start = q_end/n to avoid
    a zero‐width ramp.
    """
    # 1) Check that envelope & core each sum to 1.0
    assert abs(sum(X_env)  - 1.0) < 1e-8, "Envelope fractions must sum to 1.0"
    assert abs(sum(X_core) - 1.0) < 1e-8, "Core fractions must sum to 1.0"

    # We'll use n=1000 steps in the ramp
    n = 1000
    header = f"{n + 2} 10\n"
    lines = [header]

    # 2) First line: q=0 → envelope composition
    lines.append("0.000000   " + "   ".join(f"{x:.6f}" for x in X_env) + "\n")

    # 3) Compute q_end and raw_q_start
    q_end       = 1.0 - m_core
    raw_q_start = 1.0 - m_core - m_grad

    eps_min = 1e-8
    if raw_q_start > eps_min:
        q_start = raw_q_start
    else:
        # If raw_q_start is effectively ≤ 0, force the first ramp point at q_end/n
        q_start = q_end / float(n)

    # 4) Build n linearly spaced q‐values from q_start → q_end
    qs = np.linspace(q_start, q_end, n)

    # 5) Interpolate each of the 10 species from envelope → core over those n points
    species_vals = [np.linspace(X_env[i], X_core[i], n) for i in range(10)]

    for i in range(n):
        row_q    = f"{qs[i]:.6f}"
        row_fracs = [species_vals[j][i] for j in range(10)]
        lines.append(row_q + "   " + "   ".join(f"{x:.6f}" for x in row_fracs) + "\n")

    # 6) Final row: q=1.000000 → core composition
    lines.append("1.000000   " + "   ".join(f"{x:.6f}" for x in X_core) + "\n")

    return "".join(lines)


def write_wd_slope(filename, X_env, X_core, m_core, m_grad):
    content = conv_wd_slope(X_env, X_core, m_core, m_grad)
    with open(filename, "w") as f:
        f.write(content)


# -----------------------------------------------------------------------------
# 2) A driver that:
#    1) Creates a folder under LOGS/
#    2) Writes "basic_HeCO_composition.dat" inside that folder
#    3) Patches inlist_wd_builder so it points to that file, sets initial_mass,
#       and adds log_directory
#    4) Calls "./rn" to run WD_Builder
# -----------------------------------------------------------------------------
def build_wd_model_slope(
    envelope_fracs,
    core_fracs,
    m_core,
    m_grad,
    wd_mass,
    reset=False,
    verbose=False
):
    """
    envelope_fracs : list of 10 floats  (sum=1.0)
    core_fracs     : list of 10 floats  (sum=1.0)
    m_core         : fraction of mass that is core (0 < m_core < 1)
    m_grad         : width of linear ramp (in mass fraction)
    wd_mass        : total WD mass (M⊙) → patches initial_mass
    reset          : if True, always rebuild even if profile1.data exists
    verbose        : if True, run "./rn" with on‐screen output
    """
    # (a) Build a descriptive sub‐folder under LOGS/
    label = (
        f"{wd_mass:.3f}_Msun_"
        f"mcore={m_core:.5f}_"
        f"mgrad={m_grad:.5f}"
    )
    logdir = os.path.join("LOGS", label)
    profile_path = os.path.join(logdir, "profile1.data")

    # (b) Only rebuild if reset=True or profile1.data is missing
    if reset or (not isfile(profile_path)):
        os.makedirs(logdir, exist_ok=True)

        # (c) Write the composition file under logdir
        compo_file = os.path.join(logdir, "basic_HeCO_composition.dat")
        write_wd_slope(compo_file, envelope_fracs, core_fracs, m_core, m_grad)

        # (d) Patch inlist_wd_builder
        with open("inlist_wd_builder", "r") as f:
            text = f.read()

        # (d1) Replace relax_composition_filename
        text = re.sub(
            r"relax_composition_filename\s*=\s*'.*'",
            f"relax_composition_filename = '{compo_file}'",
            text
        )
        # (d2) Replace initial_mass
        text = re.sub(
            r"initial_mass\s*=\s*[\d.]+",
            f"initial_mass = {wd_mass:.12f}",
            text
        )
        # (d3) Insert or replace log_directory under &controls
        if re.search(r"log_directory\s*=", text):
            text = re.sub(
                r"log_directory\s*=\s*'.*'",
                f"log_directory = '{logdir}'",
                text
            )
        else:
            text = re.sub(
                r"(&controls\s*)",
                r"\1\n    log_directory = '" + logdir + "'\n",
                text
            )

        with open("inlist_wd_builder", "w") as f:
            f.write(text)

        # (e) Run WD_Builder / MESA
        if verbose:
            system("./rn")
        else:
            system("./rn &> /dev/null")

    else:
        print(f"[SKIP] WD model already exists in '{logdir}'. No rebuild.")


# -----------------------------------------------------------------------------
# 3) Example invocation with new (m_core, m_grad) so the ramp starts “farther out.”
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    #
    # Original defaults (in your old script) were:
    #   m_core = 0.998
    #   m_grad = 0.01
    # which makes the ramp confined to the innermost ∼0.2% of the star → hence “flat” to q≈0.99.
    #
    # Let’s pick instead a broader ramp that begins at q≈0.20 and ends at q≈0.40:
    #
    #   m_core = 0.60   →  core occupies inner 60% of mass  (so q_end = 0.40)
    #   m_grad = 0.20   →  ramp width = 0.20  (so raw_q_start = 0.20)
    #
    # That yields:
    #   • Pure envelope from q=0 → 0.20
    #   • Linear ramp from q=0.20 → 0.40
    #   • Pure core from q=0.40 → 1.00
    #

    envelope_fracs = [0.00, 0.00, 1.00,   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    core_fracs     = [0.00, 0.00, 0.00,   0.30, 0.00, 0.68, 0.00, 0.00, 0.02, 0.00]
    m_core         = 0.60      # 60% of the mass is “core”
    m_grad         = 0.20      # ramp spans 20% of the mass (so raw_q_start=0.20, q_end=0.40)
    wd_mass        = 0.600     # total WD mass

    build_wd_model_slope(
        envelope_fracs=envelope_fracs,
        core_fracs=core_fracs,
        m_core=m_core,
        m_grad=m_grad,
        wd_mass=wd_mass,
        reset=True,
        verbose=True
    )
