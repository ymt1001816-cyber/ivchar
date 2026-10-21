# ivchar

Automated MOSFET/FinFET I-V characterization in Python: an instrument abstraction layer
(PyVISA/SCPI driver for a Keysight B1500A or generic SMU, plus a simulated backend),
automatic parameter extraction (Vth, SS, gm, Ion/Ioff, DIBL), batch statistics across
dies/temperatures/process corners, and automatic HTML reports.

> **Note:** All example data in this repository comes from the built-in **simulated**
> compact-model backend, not from measured silicon.

*Work in progress.*
