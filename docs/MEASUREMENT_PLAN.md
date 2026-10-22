# Measurement plan: 2N7000 discrete MOSFETs

This plan takes ivchar from simulated data to **measured** data, using parts and instruments a
university teaching lab usually has: a bag of 2N7000 small-signal MOSFETs and two single-channel
SMUs (for example Keithley 2400-series units) or one multi-channel analyzer (Keysight B1500A,
Keithley 4200A).

The goal is a measured counterpart to the simulated batch. Measuring 5 to 10 parts from the same
lot gives part-to-part statistics, which run through the same extraction, outlier and report
pipeline.

## 1. Device and limits

Check every number below against the datasheet for **your** part's manufacturer before measuring.
Pinouts and ratings differ between vendors.

| Item | Typical 2N7000 datasheet value | Limit used in this plan |
|---|---|---|
| V_DS max | 60 V | ≤ 3 V |
| V_GS max | ±20 V | 0 to 4 V |
| I_D continuous | 200 mA | compliance 100 mA |
| P_D (TO-92, 25 °C) | 400 mW | ≤ 0.2 W at any bias point |
| V_GS(th) | 0.8–3.0 V at V_DS = V_GS, I_D = 1 mA | used as a sanity check |
| I_GSS | ≤ 10 nA at V_GS = ±15 V | gate compliance 10 µA |

The body is tied to the source inside the package, so the device has three terminals. A common
TO-92 pinout is 1 = source, 2 = gate, 3 = drain, viewed from the flat face with the leads down.
**Confirm it on the datasheet.**

## 2. Wiring (two single-channel SMUs)

```
 SMU A (gate)   HI ──────── G
 SMU B (drain)  HI ──────── D
 SMU A LO, SMU B LO ─────── S   (common ground; source and body are not on an SMU)
```

- Use 4-wire (remote sense) on the drain SMU if the fixture allows it. At 100 mA, 0.1 Ω of lead
  resistance drops 10 mV, which distorts the linear-region curve and gm.
- MOSFET gates are ESD-sensitive. Wear a wrist strap, keep parts in their antistatic bag, and
  short the gate to the source while inserting a part.
- Turn both outputs off before inserting or removing a part. ivchar turns every output off when a
  run ends, including when it crashes, but check the front panels anyway.

## 3. Recipe

| Sweep | Swept | Fixed | Step | Why |
|---|---|---|---|---|
| Id–Vg, linear | Vg 0 → 4 V | Vd = 0.1 V | 20 mV | Vth (max-gm), gm, SS |
| Id–Vg, saturation | Vg 0 → 4 V | Vd = 2.0 V | 20 mV | Vth (CC at 1 mA), Ion at Vg = 3 V, Ioff at Vg = 0 |
| Id–Vd family | Vd 0 → 3 V | Vg = 2.25, 2.5, 2.75, 3.0 V | 40 mV | output curves, CLM |

- **Compliance.** Drain 100 mA, gate 10 µA. Points above 3 V gate drive at Vd = 2 V will hit the
  drain limit. ivchar flags those points and excludes them from extraction.
- **Settling.** 20–50 ms per point (`--settle 0.03`), integration 1 PLC (`--nplc 1`).
- **Self-heating.** The 0.2 W cap keeps the case temperature rise small. A 2N7000's Vth shifts by
  a few mV/K, so run the high-current sweeps last and keep them short. Note the room temperature.

## 4. Procedure

1. **Dry run on the simulator** with the exact recipe, so every flag is checked before you touch
   hardware. The simulator's `discrete` preset is illustrative, not fitted to a datasheet.

   ```bash
   ivchar measure --backend sim --device discrete --vdd 3 --vg-max 4 --vd-lin 0.1 --vd-sat 2 \
     --vd-max 3 --vg-out 2.25 2.5 2.75 3.0 --step 0.02 --id-max 0.1 --ig-max 1e-5 \
     --i-crit 1e-3 --label Q-sim --out dryrun.csv
   ```

2. **Find the instruments.** Run `python -c "import pyvisa; print(pyvisa.ResourceManager().list_resources())"`.
   It lists the VISA addresses, for example `GPIB0::24::INSTR`.
3. **Open-fixture check.** Run the recipe with no part inserted. The drain current should sit at
   the instrument's noise floor (pA level). This baseline is the leakage floor for SS and Ioff.
4. **Measure each part.** Use one CSV per part, labeled Q1, Q2, ...

   ```bash
   ivchar measure --backend scpi --gate-resource GPIB0::24::INSTR --drain-resource GPIB0::25::INSTR \
     --device discrete --w 1 --l 1 --vdd 3 --vg-max 4 --vd-lin 0.1 --vd-sat 2 --vd-max 3 \
     --vg-out 2.25 2.5 2.75 3.0 --step 0.02 --id-max 0.1 --ig-max 1e-5 --settle 0.03 \
     --i-crit 1e-3 --temp 23 --label Q1 --out data/measured/2n7000_Q1.csv
   ```

5. **Sanity checks per part:** Vth (CC, 1 mA) within the datasheet's 0.8–3.0 V; gate current far
   below the 10 µA compliance; Ioff near the open-fixture floor.
6. **Report across all parts** (part-to-part variation, outliers):

   ```bash
   ivchar analyze data/measured/2n7000_Q*.csv --w 1 --l 1 --vdd 3 --i-crit 1e-3 --no-simulated \
     --label "2N7000, measured, lot <lot code>, <date>" --out report_2n7000
   ```

   With `--i-crit` set, W and L do not affect extraction, so `--w 1 --l 1` is fine for a discrete
   part.

## 5. Record keeping

For every run, record the part number and lot code, instrument models and serial numbers, the
fixture, room temperature, date, and anything unusual. Commit the raw CSVs under `data/measured/`
together with this record. Measured data stays clearly separate from simulated data, and only
`--no-simulated` reports are presented as measurements.

## 6. Known limitations

- `GenericSCPISMU` speaks the Keithley 2400 command set. A Keithley 2450 needs its 2400 emulation
  mode, and 2600-series units (TSP scripting) need a small dedicated driver.
- The drivers have been tested against expected command strings, not yet on hardware. Start with
  low compliance values and watch the first sweep.
