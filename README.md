<h1 align="center">📄 XR Twin Heart Rate Prediction Model</h1>

<p align="center">
This repository contains the <b>Heart Rate Forecasting Model</b> developed as part of the Korea IITP/MSIT-funded project:<br>
<b>"XR Twin-based Rehabilitation Training Content Technology Development"</b> (Project No. 2022-0-00218).
</p>

<hr>

<h2>🚀 Overview</h2>

<p>
This project aims to develop <b>digital therapeutics for musculoskeletal rehabilitation</b> using XR Twins, smart wearable sensors, and AI-driven coaching.<br><br>
The model predicts <b>personalized heart rate trajectories</b> during rehabilitation exercises, enabling <b>adaptive, real-time feedback</b> within XR-based rehabilitation environments.
</p>

<hr>

<h2>🎯 Project Context</h2>

<ul>
  <li><b>Project Name:</b> XR Twin-based Rehabilitation Training Content Technology Development</li>
  <li><b>Funding Agency:</b> IITP/MSIT, Republic of Korea</li>
  <li><b>Project Period:</b> 2022.04.01 ~ 2025.12.31</li>
  <li><b>Main Goal:</b> Develop XR Twin technologies for rehabilitation by integrating real-time sensor feedback, AI analysis, and customized digital coaching.</li>
  <li><b>Core Keywords:</b> Digital Therapeutics · Rehabilitation Content · XR Twin · Wearable Sensors · Personalized AI Coaching · PHR Analysis</li>
</ul>

<hr>

<h2> System Integration</h2>

<p>
The heart rate prediction model plays a critical role in the XR Twin ecosystem by:
</p>
<ul>
  <li>Processing sensor data from <b>smart wearables</b> (AR glasses, smart mirrors, IMU, EMG sensors)</li>
  <li>Forecasting <b>personalized heart rate responses</b> based on current and historical data</li>
  <li>Enabling <b>real-time adaptive coaching</b> for patients and healthcare professionals</li>
  <li>Supporting <b>PHR-based personalized healthcare services</b> within Digital Twin environments</li>
</ul>

<hr>

<h2>🧠 Model Architecture</h2>

<ul>
  <li><b>LSTM Encoder:</b> Captures personalized latent health states from historical sequences</li>
  <li><b>Adaptive Feature Selection (AdaFS):</b> Dynamically prioritizes relevant features at each time step</li>
  <li><b>Dynamic Bayesian Network (DBN):</b> Models transitions between physiological states</li>
  <li><b>Personalized Scalar Neural Networks:</b> Generates individual-specific physiological parameters (A, B, HRmin, HRmax)</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>

<ul>
  <li>Python 3.x</li>
  <li>PyTorch</li>
  <li>Scikit-learn</li>
  <li>Pandas, NumPy</li>
  <li>TQDM (for training visualization)</li>
</ul>

<hr>

<h2>📦 Repository Structure</h2>

<pre>
Model/
│   ├── dbn.py               # Main DBN Model
│   ├── modules_lstm.py      # LSTM Encoder Module
│   ├── modules_dense_nn.py  # Dense and Personalized Scalar Networks
│
Trainer/
│   ├── trainer.py           # Training pipeline and utilities
│
Data/
│   ├── data.py              # Dataset preparation and configuration
│
model_eval.ipynb             # Jupyter Notebook for training and evaluation
</pre>

<hr>

<h2>📈 Workflow</h2>

<ol>
  <li>Input workout and sensor data (historical heart rate, activity levels, IMU/EMG signals)</li>
  <li>Encode workout history using a bidirectional LSTM</li>
  <li>Apply adaptive attention using AdaFS to focus on key features</li>
  <li>Predict future physiological states using the DBN model</li>
  <li>Output personalized heart rate forecasts for AI-driven coaching</li>
</ol>

<hr>

<h2>🔥 Key Features</h2>

<ul>
  <li><b>Personalized Predictions:</b> Adapts outputs to each individual's fitness profile</li>
  <li><b>Adaptive Feature Weighting:</b> Dynamically selects the most relevant features during training</li>
  <li><b>Real-Time Forecasting:</b> Enables fast predictions suitable for XR platforms</li>
  <li><b>Wearable Sensor Integration:</b> Seamlessly processes data from AR glasses, smartwatches, and motion sensors</li>
</ul>

<hr>

<h2>🤝 Acknowledgements</h2>

<p>
This work was supported by the <b>Institute for Information & Communications Technology Promotion (IITP)</b> grant funded by the Korea government (<b>MSIT</b>) (No. 2022-0-00218, <i>XR Twin-based Rehabilitation Training Content Technology Development</i>).<br><br>
Special thanks to Professor Jaeyoung Choi, Jongsun Choi, and the research team for their support and guidance throughout the project.
</p>



