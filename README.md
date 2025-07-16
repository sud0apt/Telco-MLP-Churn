<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Telco Churn Predictor</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 2rem;
      background-color: #f9f9f9;
      color: #333;
    }
    h1, h2, h3 {
      color: #2c3e50;
    }
    code {
      background: #eee;
      padding: 2px 6px;
      border-radius: 4px;
    }
    pre {
      background: #f4f4f4;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
    }
    a {
      color: #2980b9;
    }
  </style>
</head>
<body>

  <h1>📞 Telco Churn Predictor</h1>

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

  <h2>📊 Sample Screenshot</h2>
  <p><img src="https://upload.wikimedia.org/wikipedia/en/d/d6/CelcomDigi_Logo.svg" alt="CelcomDigi Logo" width="300"/></p>

  <h2>🧠 Model Info</h2>
  <ul>
    <li>Architecture: MLP with 3 hidden layers + dropout</li>
    <li>Input: Encoded categorical + scaled numeric features</li>
    <li>Output: Softmax probabilities for churn prediction</li>
  </ul>

  <h2>📄 License</h2>
  <p>MIT License - free to use and modify.</p>

</body>
</html>
