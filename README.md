<!DOCTYPE html>
<html lang="en">
<body>

  <h1>Telco Churn Predictor</h1>
  <h3><a href="https://telco-mlp-churn-5qxoceoxyoxqyvrmboi7ev.streamlit.app/" target="_blank">
    🌐 Launch the Telco Churn App
</a></h3>

  <p>A deep learning web app that predicts customer churn for a telco company using an MLP (Multi-Layer Perceptron) trained from scratch on real-world customer data.</p>

  <h2>🚀 Features</h2>
  <ul>
    <li>Built with <strong>PyTorch</strong> and <strong>Streamlit</strong></li>
    <li>Handles both categorical and numerical inputs</li>
    <li>Interactive UI with charts and diagnostic insights</li>
    <li>Model training metrics visualization</li>
    <li>Custom embedding layers for categorical variables</li>
  </ul>

  <h2>📂 Project Structure</h2>
  <pre><code>.
├── app.py               # Streamlit frontend app
├── model.py             # PyTorch MLP model
├── dataset.py           # Dataset loader & preprocessors
├── train.py             # Training script
├── config.py            # Central configuration
├── telco-churm.csv      # Input dataset
├── mlp_telco_churn.pt   # Saved model weights
├── training_metrics.csv # Training log for charts
</code></pre>

  <h2>🛠️ Installation</h2>
  <pre><code>pip install -r requirements.txt</code></pre>

  <h2>▶️ Running the App</h2>
  <pre><code>streamlit run app.py</code></pre>

  <h2>📈 Model Training</h2>
  <pre><code>python train.py</code></pre>

  <h2>🧠 Model Info</h2>
  <ul>
    <li>Architecture: MLP with 3 hidden layers + dropout</li>
    <li>Input: Encoded categorical + scaled numeric features</li>
    <li>Output: Softmax probabilities for churn prediction</li>
  </ul>

  <h2>Dataset</h2>
  <pre><code>https://www.kaggle.com/datasets/blastchar/telco-customer-churn</code></pre>

</body>
</html>
