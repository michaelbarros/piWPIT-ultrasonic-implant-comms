# πWPIT: Ultrasonic Intra-body Wireless Power and Information Transfer

This repository contains the **simulation experiments, data, and figures** supporting our paper:

> **πWPIT: Personalized Intra-body Wireless Power and Information Transfer for Biomedical Implants**  

Authors: Rakshith Srivatsa and Sneha Hariharan

---

## 📌 Overview

We simulate and evaluate ultrasonic intra-body wireless power and information transfer (**πWPIT**) for biomedical implants under **anatomically realistic multilayer tissue profiles**.

Key goals:
- Analyze **attenuation and voltage harvesting** across patient-specific profiles.
- Co-optimize **frequency and piezoelectric plate thickness** for energy harvesting.
- Evaluate **communication channel capacity and BER** under OOK and FSK schemes.

---

## 🧪 Simulation Experiments

We align our simulations with **structured experimental analysis**:

1️⃣ **Tissue-Aware Ultrasonic Signal Propagation**  
Simulates 700 kHz ultrasonic waves across multilayered skin, fat, muscle, and bone profiles to evaluate attenuation and implant voltage output.

2️⃣ **Frequency and Thickness Co-Optimization for Energy Harvesting**  
Tunes ultrasonic frequency (within safety constraints) and computes resonance thickness of PZT-5A to maximize voltage output per patient profile.

3️⃣ **Communication Performance via Modulation Schemes**  
Implements OOK and FSK under AWGN, calculating BER across patient profiles and multiple SNRs to assess robustness under anatomical variability.

---

## 📂 Repository Structure
piWPIT-ultrasonic-implant-comms/

│

├── piwpit_simulation.py # Main annotated simulation script (BER, voltage, propagation)

├── figures/ # Auto-generated plot for the paper

├── data/ # Raw and processed data

└── README.md # This file

---

## 🚀 How to Run

1️⃣ Clone the repository:

git clone https://github.com/YourUsername/piWPIT-ultrasonic-implant-comms.git

cd piWPIT-ultrasonic-implant-comms

2️⃣ Install dependencies:

pip install numpy matplotlib seaborn scikit-learn

3️⃣ Run the simulation:

python piwpit_simulation.py

4️⃣ View generated plots:

- BER vs Patient Profiless

🩺 Scientific Details
Patient Profiles: Empirically informed anatomical layer thicknesses, density, and attenuation factors.

Ultrasound Parameters: 700–750 kHz carrier frequency, 100 MHz sampling rate.

Piezoelectric Materials: PZT-5A properties used for energy harvesting calculations.

Noise Modeling: AWGN applied with controlled SNR for realistic intra-body propagation scenarios.

📈 Outputs
The simulation outputs:

Boxplots of BER across patient profiles 

📫 Contact
For scientific questions or collaboration on ultrasonic implant communications:

Dr. Michael Taynnan Barros

University of Essex

m.barros at essex.ac.uk
