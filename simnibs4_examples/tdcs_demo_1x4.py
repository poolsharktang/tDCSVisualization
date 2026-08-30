from simnibs import sim_struct, run_simnibs


# ============================================================
# SimNIBS HD-tDCS simulation
#
# Anode:
#     POz = +2.0 mA
#
# Cathodes:
#     P1 = -0.5 mA
#     P2 = -0.5 mA
#     O1 = -0.5 mA
#     O2 = -0.5 mA
#
# Electrode diameter = 10 mm
# ============================================================


# ------------------------------------------------------------
# 1. Create simulation session
# ------------------------------------------------------------

S = sim_struct.SESSION()

S.subpath = "m2m_ernie"

S.pathfem = "tdcs_POz_4return"

# v = voltage
# e = |E|
# E = electric field vector
# j = |J|
# J = current density vector
S.fields = "veEjJ"

S.open_in_gmsh = True


# ------------------------------------------------------------
# 2. Create tDCS simulation
# ------------------------------------------------------------

tdcs = S.add_tdcslist()


# ------------------------------------------------------------
# 3. Define stimulation currents
#
# Channel 1: POz  +2.0 mA
# Channel 2: P1   -0.5 mA
# Channel 3: P2   -0.5 mA
# Channel 4: O1   -0.5 mA
# Channel 5: O2   -0.5 mA
#
# Total:
# +2.0 - 0.5 - 0.5 - 0.5 - 0.5 = 0 mA
# ------------------------------------------------------------

tdcs.currents = [
    +2.0e-3,
    -0.5e-3,
    -0.5e-3,
    -0.5e-3,
    -0.5e-3
]


# ------------------------------------------------------------
# Helper function to create a 10-mm circular electrode
# ------------------------------------------------------------

def add_electrode(tdcs, position, channel):

    electrode = tdcs.add_electrode()

    electrode.channelnr = channel

    # EEG 10-10 position
    electrode.centre = position

    # Circular electrode
    electrode.shape = "ellipse"

    # Diameter = 10 mm = 1 cm
    electrode.dimensions = [10, 10]

    # Electrode / gel thickness
    electrode.thickness = 4

    return electrode


# ------------------------------------------------------------
# 4. Add ANODE
# ------------------------------------------------------------

anode = add_electrode(
    tdcs,
    position="POz",
    channel=1
)


# ------------------------------------------------------------
# 5. Add four CATHODES
# ------------------------------------------------------------

cathode_P1 = add_electrode(
    tdcs,
    position="P1",
    channel=2
)

cathode_P2 = add_electrode(
    tdcs,
    position="P2",
    channel=3
)

cathode_O1 = add_electrode(
    tdcs,
    position="O1",
    channel=4
)

cathode_O2 = add_electrode(
    tdcs,
    position="O2",
    channel=5
)


# ------------------------------------------------------------
# 6. Run simulation
# ------------------------------------------------------------

print("Starting SimNIBS simulation...")
print()
print("Electrode configuration:")
print("  POz : +2.0 mA")
print("  P1  : -0.5 mA")
print("  P2  : -0.5 mA")
print("  O1  : -0.5 mA")
print("  O2  : -0.5 mA")
print()
print("Electrode diameter: 10 mm")

run_simnibs(S)

print()
print("Simulation finished.")
print("Results saved to:")
print(S.pathfem)