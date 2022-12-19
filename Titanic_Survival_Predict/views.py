from django.shortcuts import render
import pickle


# our home page view
def home(request):
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    model = pickle.load(open("./titanic_survival_ml_model.sav", "rb"))
    scaled = pickle.load(open("./scaler.sav", "rb"))
    prediction = model.predict(scaled.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))

    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"


def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = float(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = float(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    res = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'res': res})
