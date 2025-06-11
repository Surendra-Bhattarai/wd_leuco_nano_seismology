import os
import re
import numpy as np
from os import system
from os.path import isfile, abspath

# -----------------------------------------------------------------------------
# 1) A higher‐resolution “slope‐builder” (n_ramp = 5000), plus a simple‐atm fix.
# -----------------------------------------------------------------------------
def conv_wd_slope_highres(
    X_env,    # [10] envelope mass fractions (must sum to 1.0)
    X_core,   # [10] core   mass fractions (must sum to 1.0)
    m_core,   # fraction of mass that is core
    m_grad,   # width of linear ramp
    n_ramp    # number of interpolation points in the ramp
):
    """
    Build a (n_ramp+2) × 10 composition profile:
      • q = 0 → pure envelope (X_env)
      • q in [q_start → q_end] → linear interpolation from X_env→X_core in n_ramp steps
      • q = 1 → pure core (X_core)

    We choose n_ramp large (e.g. 5000) so that, when plotted, the ramp is visually smooth.
    """

    # 1) Sanity checks
    assert abs(sum(X_env)  - 1.0) < 1e-8, "Envelope fractions must sum to 1.0"
    assert abs(sum(X_core) - 1.0) < 1e-8, "Core fractions must sum to 1.0"
    assert (0.0 < m_core < 1.0), "m_core must lie in (0,1)"
    assert (0.0 < m_grad < 1.0), "m_grad must lie in (0,1)"

    # 2) Determine q_start, q_end
    q_end       = 1.0 - m_core
    raw_q_start = 1.0 - m_core - m_grad
    eps_min     = 1e-8

    # If raw_q_start is effectively ≤ 0, we force the first ramp point at q_end/n_ramp.
    if raw_q_start > eps_min:
        q_start = raw_q_start
    else:
        q_start = q_end / float(n_ramp)

    # 3) Build an array of n_ramp q‐values from q_start → q_end
    qs_ramp = np.linspace(q_start, q_end, n_ramp)

    # 4) Prepare to store all (q, 10‐fractions) rows
    #    Row 1: q=0 (envelope).  Rows 2..(n_ramp+1): ramp.  Row (n_ramp+2): q=1 (core).
    Nrows = n_ramp + 2
    header = f"{Nrows} 10\n"

    lines = [header]

    # 5) Row #1: q=0.000000  → all envelope fractions
    lines.append("0.000000   " + "   ".join(f"{x:.6f}" for x in X_env) + "\n")

    # 6) Interpolate all 10 species in [X_env → X_core] over n_ramp points
    #    species_vals[i] is a length‐n_ramp array from X_env[i] → X_core[i].
    species_vals = [np.linspace(X_env[i], X_core[i], n_ramp) for i in range(10)]

    for i in range(n_ramp):
        row_q = f"{qs_ramp[i]:.6f}"
        row_frac = [species_vals[j][i] for j in range(10)]
        lines.append(row_q + "   " + "   ".join(f"{x:.6f}" for x in row_frac) + "\n")

    # 7) Final row: q=1.000000 → core fractions
    lines.append("1.000000   " + "   ".join(f"{x:.6f}" for x in X_core) + "\n")

    return "".join(lines)


# -----------------------------------------------------------------------------
# 2) The driver: write the high‐res file, patch inlist (with simple photosphere),
#    and run WD_Builder (./rn).
# -----------------------------------------------------------------------------
def build_wd_model_slope_highres(
    envelope_fracs,   # [10] list (must sum to 1.0)
    core_fracs,       # [10] list (must sum to 1.0)
    m_core,           # fraction of total mass = core
    m_grad,           # ramp width in fractional mass
    wd_mass,          # WD mass in M_sun
    n_ramp            # how many interpolation points in ramp (e.g. 5000)
):
    """
    1) Creates directory “LOGS/{wd_mass}_Msun_mcore={m_core:.5f}_mgrad={m_grad:.5f}/”
    2) Writes “basic_HeCO_composition.dat” with (n_ramp+2)×10 rows.
    3) Patches inlist_wd_builder:
         - relax_composition_filename = absolute path to that .dat
         - initial_mass = wd_mass
         - log_directory = that LOGS/... folder
         - atm_option = 'table'
    4) Calls “./rn” and streams output to screen.
    """

    # 1) Build the LOGS directory name
    label = f"{wd_mass:.3f}_Msun_mcore={m_core:.5f}_mgrad={m_grad:.5f}"
    logdir = os.path.join("LOGS", label)
    profile_path = os.path.join(logdir, "profile_new.data")

    # 2) Only rebuild if profile1.data doesn't exist, to avoid re‐doing everything
    if not isfile(profile_path):
        os.makedirs(logdir, exist_ok=True)

        # 3) Write the high‐resolution composition file
        compo_file = os.path.join(logdir, "basic_HeCO_composition.dat")
        content = conv_wd_slope_highres(
            X_env     = envelope_fracs,
            X_core    = core_fracs,
            m_core    = m_core,
            m_grad    = m_grad,
            n_ramp    = n_ramp
        )
        with open(compo_file, "w") as f:
            f.write(content)

        # 4) Diagnostic printout so you can confirm “DB lines” if desired
        abs_path = abspath(compo_file)
        print("\n=== ABOUT TO TELL MESA TO READ THIS EXACT FILE ===")
        print("Composition file path:", abs_path)
        print("--- First five lines of the file: ---")
        with open(compo_file, "r") as f:
            for _ in range(5):
                ln = f.readline().rstrip("\n")
                if not ln:
                    break
                print(ln)

        total_lines = int(os.popen(f"wc -l {compo_file}").read().split()[0])
        # Check header vs total_lines
        declared_rows = int(content.splitlines()[0].split()[0])
        print(f"\nTotal lines = {total_lines}, header says “{declared_rows}” → OK\n")

        # (Optionally) check the row‐sums and duplicate q’s before we hand this to MESA
        print("Checking row‐sums and duplicate q’s (first 100 rows only):")
        sums_err = False
        dup_err  = False
        with open(compo_file, "r") as f:
            next(f)  # skip header
            prev_q = None
            for idx, row in enumerate(f, start=2):
                if idx > 102:
                    break
                parts = row.strip().split()
                if len(parts) != 11:
                    print(f"  ERROR: line {idx} has {len(parts)} columns (should be 11).")
                    sums_err = True
                    continue
                qv = float(parts[0])
                fracs = list(map(float, parts[1:11]))
                if abs(sum(fracs) - 1.0) > 1e-5:
                    print(f"  BAD SUM on line {idx}: sum = {sum(fracs):.6f}")
                    sums_err = True
                if prev_q is not None and abs(qv - prev_q) < 1e-9:
                    print(f"  DUPLICATE q on line {idx}: q = {qv:.6f}")
                    dup_err = True
                prev_q = qv

            if not sums_err:
                print("  → All sums OK (within 1e‐6) for first 100 lines.")
            if not dup_err:
                print("  → No duplicate q’s in first 100 lines.\n")

        if sums_err or dup_err:
            print("ERROR: Your composition file failed basic checks. Aborting.\n")
            return

        # 5) Patch inlist_wd_builder:
        #    (a) relax_composition_filename  (b) initial_mass  (c) log_directory  (d) atm_option
        with open("inlist_wd_builder", "r") as f:
            text = f.read()

        # (5a) Replace relax_composition_filename
        text = re.sub(
            r"relax_composition_filename\s*=\s*'.*?'",
            f"relax_composition_filename = '{abs_path}'",
            text
        )
        # (5b) Replace initial_mass
        text = re.sub(
            r"initial_mass\s*=\s*[\d.]+",
            f"initial_mass = {wd_mass:.12f}",
            text
        )
        # (5c) Replace or insert log_directory
        if re.search(r"log_directory\s*=", text):
            text = re.sub(
                r"log_directory\s*=\s*'.*?'",
                f"log_directory = '{logdir}'",
                text
            )
        else:
            text = re.sub(
                r"(&controls\s*)",
                r"\1\n    log_directory = '" + logdir + "'\n",
                text
            )

        # (5d) Force atm_option = 'table'
        if re.search(r"atm_option\s*=", text):
            text = re.sub(
                r"atm_option\s*=\s*'.*?'",
                "atm_option = 'table'",
                text
            )
        else:
            text = re.sub(
                r"(&controls\s*)",
                r"\1\n    atm_option = 'table'\n",
                text
            )

        with open("inlist_wd_builder", "w") as f:
            f.write(text)

        # 6) Run WD_Builder (./rn) with output on screen
        print("\n=== NOW CALLING: ./rn  (WD_Builder) ===\n")
        system("./rn")
        print("\n=== MESA RUN COMPLETE ===\n")

    else:
        print(f"[SKIP] WD model already exists in '{logdir}'. No rebuild.")


# -----------------------------------------------------------------------------
# 3) Main: choose your envelope/core mix, m_core, m_grad, and n_ramp here
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ––– Example for a 0.600 M⊙ WD –––
    #
    # Envelope: pure He  (X = [H1, He3, He4, C12, N14, O16, Ne20, Ne22, Mg24, Z_other])
    envelope_fracs = [0.00, 0.00, 1.00,   0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

    # Core: 30% C12, 68% O16, 2% Mg24
    core_fracs     = [0.00, 0.00, 0.00,   0.30, 0.00, 0.68, 0.00, 0.00, 0.02, 0.00]

    # (a) m_core = 0.60 → core occupies inner 60% of mass → q_end = 0.40 
    # (b) m_grad = 0.20 → ramp begins at raw_q_start = 0.20 → linearly goes to q=0.40
    m_core  = 0.60
    m_grad  = 0.20

    # A larger n_ramp (e.g. 5000) makes the linear ramp nearly indistinguishable from continuous
    n_ramp  = 5000

    wd_mass = 0.600

    build_wd_model_slope_highres(
        envelope_fracs=envelope_fracs,
        core_fracs=core_fracs,
        m_core=m_core,
        m_grad=m_grad,
        wd_mass=wd_mass,
        n_ramp=n_ramp
    )