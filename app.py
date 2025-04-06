from flask import Flask, render_template, request
from regresi import load_and_train_model, predict_temperature, generate_plot

app = Flask(__name__)

# Load model saat server mulai
model, scaler_x, scaler_y = load_and_train_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = None

    if request.method == 'POST':
        try:
            humidity = float(request.form['humidity'])
            prediction = predict_temperature(humidity, model, scaler_x, scaler_y)
            plot_url = generate_plot(model, scaler_x, scaler_y)
        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template('index.html', prediction=prediction, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)

