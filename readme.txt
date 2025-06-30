I've uploaded a script and two VGM files that failed to process correctly. The script extracts unique FM instruments and DAC sample groups by hashing register state (e.g., operator TL, AR/DR, FMS, etc.), logging key on/off events, and outputting new .vgms per instrument or sample group with preserved timing.

Test these two VGM files (`AIZ1.vgm`, `FFZ.vgm`) with the script. Analyze why instrument isolation failed, and suggest fixes or limitations in the current logic that could be addressed. Summary of expected behavior is below:

- Instruments: Defined by YM2612 register combinations, grouped by hash
- DAC: Grouped by amplitude and variance (e.g. loud kicks, quiet hats)
- Output: .vgm per sound, with original structure intact
- Summary: Usage stats per instrument/sample group

Please help debug and improve compatibility for these cases.