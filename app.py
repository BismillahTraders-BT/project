from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file,encoding='utf-8')
        test_data = df.values
        predictions = model.predict(test_data)
        return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
