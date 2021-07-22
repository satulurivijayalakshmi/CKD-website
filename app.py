import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model\model.pkl', 'rb'))



@app.route('/')
def form():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    int_features=[]
    
    age=request.form['Age']
    int_features.append(float(age))
    bp=request.form['Blood Pressure']
    int_features.append(float(bp))
    sp=request.form['Specific Gravity']
    int_features.append(float(sp))
    al=request.form['Albumin']
    int_features.append(float(al))
    su=request.form['Sugar']
    int_features.append(float(su))
    rbc=request.form['rbc']
    int_features.append(float(rbc))
    pc=request.form['pc']
    int_features.append(float(pc))
    pcc=request.form['pcc']
    int_features.append(float(pcc))
    bac=request.form['bac']
    int_features.append(float(bac))
    bgr=request.form['Blood Glucose Random']
    int_features.append(float(bgr))
    bu=request.form['Blood Urea']
    int_features.append(float(bu))
    sc=request.form['Serum Createinine']
    int_features.append(float(sc))
    so=request.form['Sodium']
    int_features.append(float(so))
    po=request.form['Potassium']
    int_features.append(float(po))
    hemo=request.form['Hemoglobin']
    int_features.append(float(hemo))
    pcv=request.form['Packed Cell Volume']
    int_features.append(float(pcv))
    wbcc=request.form['White Blood Cells Count']
    int_features.append(float(wbcc))
    rbcc=request.form['Red Blood Cells Count']
    int_features.append(float(rbcc))
    hptn=request.form['hptn']
    int_features.append(float(hptn))
    dm=request.form['dm']
    int_features.append(float(dm))
    cad=request.form['cad']
    int_features.append(float(cad))
    ap=request.form['ap']
    int_features.append(float(ap))
    pe=request.form['pe']
    int_features.append(float(pe))
    ane=request.form['ane']
    int_features.append(float(ane))
    #print(int_features)
    final_features = [np.array(int_features)]
    #print(final_features)
    prediction = model.predict(final_features)
    #print(prediction)

    output = int(prediction)
    #print(output)
    
    if output == 0:
        return render_template('index.html', prediction_text='you are tested negative for CKD ')
    else:
        return render_template('index.html', prediction_text='you are tested positive for CKD ' )


        
if __name__=="__main__":
    app.run(debug=True)
