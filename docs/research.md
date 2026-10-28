# Background and design notes

This document records the literature review that shaped Respira's design, the
dataset details that matter for implementation, and the reasoning behind each
processing choice. It is written so the project can be defended as a piece of
biomedical signal-processing work, not just run.

## 1. Why respiration is visible in the PPG

The photoplethysmogram measures light absorbed by blood in the tissue under the
sensor. Breathing modulates the pulse waveform through three mechanisms, which
the literature names consistently (Karlen et al. 2013; Charlton et al. 2016):

| Variation | Abbreviation | Physiological origin | Measured as |
|---|---|---|---|
| Baseline wander | BW / RIIV (respiratory-induced intensity variation) | Intrathoracic pressure swings change venous return and tissue blood volume, moving the DC level of the signal | Mean of the trough and peak amplitude per beat |
| Amplitude modulation | AM / RIAV | Inspiration lowers left-ventricular stroke volume (and sympathetic tone changes vascular compliance), so pulse amplitude falls | Peak minus trough amplitude per beat |
| Frequency modulation | FM / RIFV | Respiratory sinus arrhythmia: heart rate rises during inspiration and falls during expiration | Peak-to-peak interval per beat |

All three are weak compared with the cardiac pulse, are not present with equal
strength in every person, and depend on posture, ventilation mode and disease.
That is why algorithms estimate RR from each one and then fuse.

## 2. The BIDMC PPG and Respiration Dataset

* Source: PhysioNet, https://physionet.org/content/bidmc/1.0.0/ (Open Data
  Commons Attribution License v1.0).
* Origin: 53 recordings extracted from the MIMIC-II matched waveform database,
  critically ill adults in the Beth Israel Deaconess Medical Center ICU. First
  published with Pimentel et al. (2017).
* Each recording is 8 minutes (60 001 samples at 125 Hz) and contains PPG
  (`PLETH`), thoracic impedance respiration (`RESP`) and one or more ECG leads.
  The ECG lead set varies between subjects (II, V, AVR, I, MCL, plus occasional
  ABP/CVP/ART channels), so the loader selects columns by name.
* Monitor numerics at 1 Hz: `HR`, `PULSE`, `RESP` (the monitor's own
  impedance-derived respiratory rate) and `SpO2`.
* Breath annotations: two annotators independently marked each breath on the
  impedance waveform. They are stored as signal sample numbers in two columns of
  unequal length (`bidmc_NN_Breaths.csv`), so the columns are ragged and must be
  parsed independently.
* Quirks found while loading: one Signals file (subject 51) contains a bare
  `-` placeholder in a waveform cell. The loader coerces such cells to NaN and
  linearly interpolates across them.

The ICU setting matters. Compared with the healthy volunteers used in most
validation studies, these patients have arrhythmias, mechanical ventilation,
vasoactive drugs, motion and poor perfusion, all of which degrade one or more of
the respiratory modulations. Published PPG-based results on BIDMC are therefore
markedly worse than on CapnoBase or similar healthy-subject datasets.

## 3. Published results to compare against

| Study | Data | Method | Result (32 s windows unless stated) |
|---|---|---|---|
| Karlen et al. 2013 | CapnoBase (healthy, anaesthetised) | Smart Fusion of RIIV/RIAV/RIFV | RMSE ~ 3 bpm with ~ 10 % of windows discarded |
| Charlton et al. 2016 | 39 healthy adults (Vortal) | 314 algorithm variants; best PPG: BW/AM/FM + time-domain breath detection + Smart Fusion | bias 1.0, 2SD 6.2 bpm, 95 % LoA -5.1 to 7.2, CP2 71.5 %, estimates on ~ 72 % of windows |
| Pimentel et al. 2017 | CapnoBase and BIDMC | Multiple AR model orders, pole-magnitude selection, fusion | median MAE 1.5 bpm (CapnoBase) vs 4.0 bpm, IQR 1.8-5.5 (BIDMC); > 90 % of windows retained |
| Computing in Cardiology 2021 preprint | BIDMC and CapnoBase | Direct spectral estimate from baseline-wander signal, 60 s windows | MAE 1.42 bpm |

The headline expectation for an honest 32 s evaluation on BIDMC is therefore an
MAE in the 2-5 bpm range, with ICU monitor impedance itself far from perfect.

## 4. Pipeline design

### 4.1 Filtering and beat detection

* The PPG is lightly band-passed (0.067-8 Hz, i.e. 4 bpm to 8 Hz) for feature
  measurement. The high-pass corner of 4 bpm follows Charlton et al. (2016) and
  preserves baseline wander, which is itself a respiratory signal.
* Beat detection runs on a second copy band-passed at 0.5-8 Hz so that baseline
  wander does not bias local-maximum detection. Peaks are found with a minimum
  spacing of 60/180 s and a prominence threshold of 30 % of the median prominence,
  then refined within ±60 ms on the lightly filtered signal. Troughs are the
  minima between consecutive peaks. Beats whose interval falls outside 40-180
  beats/min are removed.

### 4.2 Respiratory-induced variations

BW, AM and FM are measured per beat as in the table above, linearly
interpolated onto a 4 Hz grid (Karlen et al. 2013 use 4 Hz; Charlton et al. use
5 Hz) and band-passed to 4-60 breaths/min. A fourth, beat-free source (`filt`)
simply band-passes the raw PPG to the same band; it is the simplest possible
approach and surprisingly competitive in the literature.

### 4.3 Estimators

| Name | Family | Notes |
|---|---|---|
| `periodogram` | Frequency domain | Hann-windowed, zero-padded FFT over the whole window; RR is the largest in-band peak |
| `welch` | Frequency domain | Two 50 %-overlapping segments; lower variance, coarser resolution |
| `ar` | Frequency domain (parametric) | Burg autoregressive model (order 8); the pole nearest the peak gives a confidence (Pimentel et al. 2017) |
| `autocorr` | Time domain | First autocorrelation maximum in the plausible lag range |
| `count` | Time domain | "Count-orig" breath counting of Schäfer & Kratky (2008): maxima above 20 % of the upper-quartile maximum, separated by a sub-zero minimum |

### 4.4 Fusion

* **Smart Fusion** (`sf`): mean of the BW, AM and FM estimates when their
  standard deviation is at most 4 bpm; otherwise no output. Trades coverage for
  accuracy.
* **Median** (`median`): robust to one outlying modulation, always outputs.
* **Spectral averaging** (`spectral`): normalised periodograms of the three RIVs
  are averaged before peak picking, so a frequency supported by two modulations
  wins even if a third is noisy.
* **Temporal smoothing** (`sf_ts`): exponential smoothing across consecutive
  windows (Charlton et al. 2016, method FT1: 0.2 new + 0.8 previous), carrying
  the last estimate forward through rejected windows.

### 4.5 Signal quality

A template-matching SQI in the style of Orphanidou et al. (2015): each pulse
(-250 ms to +400 ms around the peak) is correlated with the window's mean
pulse; the SQI is the mean correlation. A window is also checked for plausible
heart rate (40-180), no gap longer than 3 s and a max/min beat-interval ratio
below 2.2. The evaluation reports accuracy against SQI bins and sweeps the
acceptance threshold to show the coverage/accuracy trade-off.

### 4.6 Reference respiratory rate

For each 32 s window the RR of each annotator is 60 × (breaths − 1) / (time
between first and last breath in the window). The reference is the mean of
the two annotators, and the window is excluded when they disagree by more than
2 bpm or either marks fewer than three breaths. Two further comparators are
scored against the same reference: the monitor's own impedance RR numeric and
a spectral estimate from the raw impedance waveform.

### 4.7 Metrics

Mean absolute error (primary), median absolute error, RMSE, bias, 95 % limits
of agreement (bias ± 1.96 SD), coverage probability within 2 bpm (CP2) and
coverage (fraction of reference windows for which the algorithm produced an
estimate). Coverage matters because an algorithm can look accurate simply by
abstaining on hard windows.

## 5. References

* Charlton, P.H., Bonnici, T., Tarassenko, L., Clifton, D.A., Beale, R.,
  Watkinson, P.J. (2016). An assessment of algorithms to estimate respiratory
  rate from the electrocardiogram and photoplethysmogram. *Physiological
  Measurement* 37(4), 610-626. https://doi.org/10.1088/0967-3334/37/4/610
* Charlton, P.H. et al. (2018). Breathing rate estimation from the
  electrocardiogram and photoplethysmogram: a review. *IEEE Reviews in
  Biomedical Engineering* 11, 2-20.
* Karlen, W., Raman, S., Ansermino, J.M., Dumont, G.A. (2013). Multiparameter
  respiratory rate estimation from the photoplethysmogram. *IEEE Transactions
  on Biomedical Engineering* 60(7), 1946-1953.
* Orphanidou, C. et al. (2015). Signal-quality indices for the electrocardiogram
  and photoplethysmogram: derivation and applications to wireless monitoring.
  *IEEE Journal of Biomedical and Health Informatics* 19(3), 832-838.
* Pimentel, M.A.F., Johnson, A.E.W., Charlton, P.H., Birrenkott, D.,
  Watkinson, P.J., Tarassenko, L., Clifton, D.A. (2017). Toward a robust
  estimation of respiratory rate from pulse oximeters. *IEEE Transactions on
  Biomedical Engineering* 64(8), 1914-1923.
  https://doi.org/10.1109/TBME.2016.2613124
* Schäfer, A., Kratky, K.W. (2008). Estimation of breathing rate from
  respiratory sinus arrhythmia: comparison of various methods. *Annals of
  Biomedical Engineering* 36(3), 476-485.
* Goldberger, A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation* 101(23), e215-e220.
