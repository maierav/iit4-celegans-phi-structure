# Simbrain export

`*_effective_weights.csv` — the 4x4 effective-sensitivity matrices (rows = source
at t, cols = target at t+1) for the core and sensory quartets, from the baseline
TPM. `*_tpm_*.csv` — the smoothed 16x16 conditioned TPMs (state labels
`format(int,'04b')`, leftmost bit = 4th neuron of the quartet).

`build_quartet.bsh` — a Simbrain 3.x script-console starter that builds the
quartet with these weights. **Untested against a live Simbrain install**; treat
as a template. Two semantic caveats are in its header: the weights are unsigned
coupling strengths, and Simbrain's continuous neurons are not the binarized
Markov chain — the faithful simulator is `../interactive/quartet_simulator.html`,
which runs the conditioned TPMs directly and overlays the recorded PSTHs.
