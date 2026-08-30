from simnibs import sim_struct, run_simnibs


# ============================================================
# HD-tDCS montage
#
# ANODES:
#   C3  = +1.0 mA
#   C4  = +1.0 mA
#
# CATHODES:
#   FC3 = -0.3333 mA
#   FC4 = -0.3333 mA
#   C5  = -0.3333 mA
#   C6  = -0.3333 mA
#   CP3 = -0.3333 mA
#   CP4 = -0.3333 mA
#
# Total stimulation current = 2 mA
# Electrode diameter = 10 mm
# ============================================================


# ------------------------------------------------------------
# 1. Create session
# ------------------------------------------------------------

S = sim_struct.SESSION()

S.subpath = "m2m_ernie"

S.pathfem = "tdcs_C3_C4_multicathode"

S.fields = "veEjJ"

S.open_in_gmsh = True

# Optional: create volumetric results for MRI-style slices
S.map_to_vol = True


# ------------------------------------------------------------
# 2. Create TDCS simulation
# ------------------------------------------------------------

tdcs = S.add_tdcslist()


# ------------------------------------------------------------
# 3. Define currents
#
# 2 anodes:
#    +1 mA each
#
# 6 cathodes:
#    -2 mA / 6 = -0.333333 mA each
# ------------------------------------------------------------

tdcs.currents = [
    +1.0e-3,           # Channel 1: C3
    +1.0e-3,           # Channel 2: C4

    -2.0e-3 / 6,       # Channel 3: FC3
    -2.0e-3 / 6,       # Channel 4: FC4
    -2.0e-3 / 6,       # Channel 5: C5
    -2.0e-3 / 6,       # Channel 6: C6
    -2.0e-3 / 6,       # Channel 7: CP3
    -2.0e-3 / 6        # Channel 8: CP4
]


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def add_electrode(tdcs, position, channel):

    electrode = tdcs.add_electrode()

    electrode.channelnr = channel

    # EEG 10-10 position
    electrode.centre = position

    # Circular electrode
    electrode.shape = "ellipse"

    # 1 cm diameter = 10 mm
    electrode.dimensions = [10, 10]

    # Electrode / gel thickness
    electrode.thickness = 4

    return electrode


# ------------------------------------------------------------
# 4. Add ANODES
# ------------------------------------------------------------

add_electrode(tdcs, "C3", 1)
add_electrode(tdcs, "C4", 2)


# ------------------------------------------------------------
# 5. Add CATHODES
# ------------------------------------------------------------

add_electrode(tdcs, "FC3", 3)
add_electrode(tdcs, "FC4", 4)

add_electrode(tdcs, "C5", 5)
add_electrode(tdcs, "C6", 6)

add_electrode(tdcs, "CP3", 7)
add_electrode(tdcs, "CP4", 8)


# ------------------------------------------------------------
# 6. Run simulation
# ------------------------------------------------------------

print("Starting SimNIBS simulation...")
print()
print("ANODES:")
print(" C3  = +1.000 mA")
print(" C4  = +1.000 mA")

print()
print("CATHODES:")
print(" FC3 = -0.333 mA")
print(" FC4 = -0.333 mA")
print(" C5  = -0.333 mA")
print(" C6  = -0.333 mA")
print(" CP3 = -0.333 mA")
print(" CP4 = -0.333 mA")

print()
print("Electrode diameter = 10 mm")
print("Total stimulation current = 2 mA")

run_simnibs(S)

print("Simulation finished.")