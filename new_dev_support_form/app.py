from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle form submission
        data = request.form
        # Process the data as needed
        return jsonify({"message": "Form submitted successfully", "data": data})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
