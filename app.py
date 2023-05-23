import pandas as pandas
from flask import Flask, request
import joblib
import numpy

MODEL_PATH = 'mlmodels/model.pkl'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.route('/predict_price', methods = ['GET'])
def predict():
    args = request.args
    open_plan = args.get('open_plan', default=-1, type=int)
    rooms = args.get('rooms', default=-1, type=int)
    area = args.get('area', default=-1, type=float)
    renovation = args.get('renovation', default=-1, type=float)
    floor = args.get('floor', default=-1, type=int)
    studio = args.get('studio', default=-1, type=int)
    offer_time = args.get('offer_time', default=-1, type=int)
    agent_fee = args.get('agent_fee', default=-1, type=float)

    x = numpy.array([open_plan, rooms, int(area), int(renovation), floor, studio, offer_time, int(agent_fee)]).reshape(1, -1)
    result = model.predict(x)

    return str(result[0][0])


if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')

