#!/usr/bin/env python3
import os
import re
import numpy as np
from os import system
from os.path import isfile

# -----------------------------------------------------------------------------
# 1) Build a “(n+2)×10” composition file with a linear ramp (n=100),
#    ensuring no two q‐values are effectively identical by requiring
#    raw_q_start > ε_min. If raw_q_start ≤ ε_min, set q_start = q_end/n.
#
#    Column order (10 species):
#      [ H1, He3, He4, C12, N14, O16, Ne20, Ne22, Mg24, Z_other ]
# -----------------------------------------------------------------------------

def conv_wd_slope(
    X_env,    # length‐10 list: envelope mass fractions (sum to 1.0)
    X_core,   # length‐10 list: core mass fractions   (sum to 1.0)
    m_core,   # core mass fraction (so envelope is q ∈ [0, 1−m_core])
    m_grad    # width of the linear ramp (in mass fraction)
):
    """
    Returns a string for a composition file with:
      • Row 1: q = 0.000000          → envelope composition (X_env)
      • Rows 2..(n+1): n=100 lines,
        if raw_q_start = 1−m_core−m_grad > ε_min, ramp from raw_q_start → q_end=1−m_core,
        otherwise set q_start = q_end/n so the first ramp point is q_end/100.
      • Last row: q = 1.000000        → core composition (X_core)

    Total rows = n + 2 = 102  → header “102 10”.
    Each row has 11 numbers: q + 10 mass fractions.
    """

    # 1) Sanity checks: each composition vector must sum to 1.0
    assert abs(sum(X_env)  - 1.0) < 1e-8, "Envelope fractions must sum to 1.0"
    assert abs(sum(X_core) - 1.0) < 1e-8, "Core fractions must sum to 1.0"

    n = 1000
    header = f"{n + 2} 10\n"
    lines = [header]

    # Row 1: surface, q=0, pure envelope
    lines.append("0.000000   " + "   ".join(f"{x:.6f}" for x in X_env) + "\n")

    # 2) Compute q_end and raw_q_start
    q_end       = 1.0 - m_core
    raw_q_start = 1.0 - m_core - m_grad

    # Define a minimum threshold below which we treat raw_q_start as "effectively zero"
    eps_min = 1e-8

    if raw_q_start > eps_min:
        q_start = raw_q_start
    else:
        # Force first ramp‐point to be q_end / n (so > 0)
        q_start = q_end / float(n)

    # Build n linearly spaced q-values from q_start → q_end
    qs = np.linspace(q_start, q_end, n)

    # 3) Interpolate each of the 10 species from envelope → core in n steps
    species_vals = [np.linspace(X_env[i], X_core[i], n) for i in range(10)]

    for i in range(n):
        row_q    = f"{qs[i]:.6f}"
        row_fracs = [species_vals[j][i] for j in range(10)]
        lines.append(row_q + "   " + "   ".join(f"{x:.6f}" for x in row_fracs) + "\n")

    # 4) Final row at q=1.0: core composition
    lines.append("1.000000   " + "   ".join(f"{x:.6f}" for x in X_core) + "\n")

    return "".join(lines)


def write_wd_slope(filename, X_env, X_core, m_core, m_grad):
    """
    Write the slope composition string to `filename`.
    """
    content = conv_wd_slope(X_env, X_core, m_core, m_grad)
    with open(filename, "w") as f:
        f.write(content)


# -----------------------------------------------------------------------------
# 2) A driver that:
#    1) Creates a unique folder under LOGS/
#    2) Writes "basic_HeCO_composition.dat" inside that folder using conv_wd_slope
#    3) Patches `inlist_wd_builder` so it points to that file, sets initial_mass,
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
    Parameters
    ----------
    envelope_fracs : list of 10 floats
        Mass fractions [H1, He3, He4, C12, N14, O16, Ne20, Ne22, Mg24, Z_other]
        in the envelope (must sum to 1.0).

    core_fracs : list of 10 floats
        Mass fractions in the core (must sum to 1.0).

    m_core : float
        Mass fraction of the WD that is "core". (Envelope extends from q=0 to q=1 - m_core.)

    m_grad : float
        Width of the linear compositional ramp (in mass fraction).

    wd_mass : float
        Total WD mass in M_sun, used to patch `initial_mass` in the inlist.

    reset : bool
        If True, force a rebuild even if LOGS/<label>/profile1.data exists.

    verbose : bool
        If True, run "./rn" without redirecting output.
    """

    # 1) Construct a descriptive label and directory under LOGS/
    label = (
        f"{wd_mass:.3f}_Msun_"
        f"mcore={m_core:.5f}_"
        f"mgrad={m_grad:.5f}"
    )
    logdir = os.path.join("LOGS", label)
    profile_path = os.path.join(logdir, "profile1.data")

    # 2) Check if we need to build
    if reset or (not isfile(profile_path)):
        os.makedirs(logdir, exist_ok=True)

        # 3) Write the 102×10 composition file under logdir
        compo_file = os.path.join(logdir, "basic_HeCO_composition.dat")
        write_wd_slope(compo_file, envelope_fracs, core_fracs, m_core, m_grad)

        # 4) Patch inlist_wd_builder
        with open("inlist_wd_builder", "r") as f:
            text = f.read()

        # 4a) Replace relax_composition_filename
        text = re.sub(
            r"relax_composition_filename\s*=\s*'.*'",
            f"relax_composition_filename = '{compo_file}'",
            text
        )
        # 4b) Replace initial_mass
        text = re.sub(
            r"initial_mass\s*=\s*[\d.]+",
            f"initial_mass = {wd_mass:.12f}",
            text
        )
        # 4c) Insert or replace log_directory under &controls
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

        # 5) Run WD_Builder / MESA
        if verbose:
            system("./rn")
        else:
            system("./rn &> /dev/null")
    else:
        print(f"[SKIP] WD model already exists in '{logdir}'. No rebuild.")


# -----------------------------------------------------------------------------
# 3) Example invocation (uncomment to run directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters for a 0.600 M_sun WD:
    #  • Envelope: pure He
    #  • Core: 30% C12, 68% O16, 2% Mg24
    #  • m_core = 0.998 → core occupies inner 99.8% of mass
    #  • m_grad = 0.002 → 0.2% mass‐wide linear ramp from He → CO+Mg
    envelope_fracs = [0.00, 0.00, 1.00,   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    core_fracs     = [0.00, 0.00, 0.00,   0.30, 0.00, 0.68, 0.00, 0.00, 0.02, 0.00]
    m_core         = 0.998
    m_grad         = 0.01
    wd_mass        = 0.600

    build_wd_model_slope(
        envelope_fracs=envelope_fracs,
        core_fracs=core_fracs,
        m_core=m_core,
        m_grad=m_grad,
        wd_mass=wd_mass,
        reset=True,
        verbose=True
    )
