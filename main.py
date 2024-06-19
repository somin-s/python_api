from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd 
import joblib
import numpy as np
# import altair as alt
# import matplotlib.pyplot as plt
# import plotly.express as px


app = Flask(__name__)
CORS(app)

model_1 = joblib.load('model1_final.joblib')
model_2 = joblib.load('model2_final.joblib')
model_3 = joblib.load('model3_final.joblib')
model_4 = joblib.load('model4_final.joblib')

@app.route("/ModelClassify", methods=['POST'])
def display():
    data = request.data
    string = data.decode('UTF-8')
    data = eval(string)

    Cluster_number_input = pd.to_numeric(data['Cluster_number'])
    Cluster_weight_input = pd.to_numeric(data['Cluster_weight'])
    Shoot_num_input = pd.to_numeric(data['Shoot_number_more_5mm'])
    Vine_canopy_input = pd.to_numeric(data['Vine_canopy'])
    Leaf_area_input = pd.to_numeric(data['Leaf_Area_per_m'])
    Berry_weight_input = pd.to_numeric(data['Berry_weight'])

    Cluster_number_ran = np.random.normal(Cluster_number_input,4.0,20)
    Cluster_weight_ran = np.random.normal(Cluster_weight_input,4.0,20)
    Shoot_number_more_5mm_ran = np.random.normal(Shoot_num_input,4.0,20)
    Vine_canopy_ran = np.random.normal(Vine_canopy_input,0.1,20)
    Leaf_Area_per_m_ran = np.random.normal(Leaf_area_input,13.6,20)
    Berry_weight_ran = np.random.normal(Berry_weight_input,0.1,20)

    Quality_yield = []

    for i in range(20):
        #Preprocess for Module1
        Cluster_number1 = np.log(Cluster_number_ran[i]/10+1)
        Cluster_weight1 = np.log(Cluster_weight_ran[i]/50+1)
        Shoot_number_gt_5mm1 = np.log(Shoot_number_more_5mm_ran[i]/10+1)
        Berry_weight1 = np.log(Berry_weight_ran[i]/10+5)
        #Output first module
        FirstMol_value = model_1.predict([[Cluster_number1, Cluster_weight1, Shoot_number_gt_5mm1,Berry_weight1]])
        #OutputFirstModule(FirstMol_value[0])

        #Preprocess for Module2
        Cluster_number2 = np.log(Cluster_number_ran[i]/10+1)
        Cluster_weight2 = np.log(Cluster_weight_ran[i]/20+20)
        Shoot_number_gt_5mm2 = np.log(Shoot_number_more_5mm_ran[i]/10+1)
        Vine_canopy2 = np.log(Vine_canopy_ran[i]+2)
        Leaf_Area_per_m2 = np.log(Leaf_Area_per_m_ran[i]/1000+10)
        Berry_weight2 = np.log(Berry_weight_ran[i]+2) 

        #Output second & third module
        SecondMol_value = model_2.predict([[Cluster_number2, Cluster_weight2, Shoot_number_gt_5mm2,Vine_canopy2,Leaf_Area_per_m2,Berry_weight2]])

        cn = round(Cluster_number_ran[i])
        cw = round(Cluster_weight_ran[i])
        sn = round(Shoot_number_more_5mm_ran[i])
        vc = round(Vine_canopy_ran[i],2)
        la = round(Leaf_Area_per_m_ran[i])
        bw = round(Berry_weight_ran[i],2)

        source = Modules(FirstMol_value[0],SecondMol_value[0])

        row1 = {'Quality':source[0], 'Yield': "Yield per wine", 'Value':source[1],
                'Info':"Information",'Cluster number':cn,'Cluster weight (g)':cw,
                'Shoot number': sn,'Vine canopy (%)': vc, 'Leaf area / metre':la,'Berry_weight_g':bw}
        row2 = {'Quality':source[0], 'Yield': "Yield per metre", 'Value':source[2],
                'Info':"Information",'Cluster number':cn,'Cluster weight (g)':cw,
                'Shoot number': sn,'Vine canopy (%)': vc, 'Leaf area / metre':la,'Berry_weight_g':bw}
        row3 = {'Quality':source[0], 'Yield': "Yield per square metre", 'Value':source[3],
                'Info':"Information",'Cluster number':cn,'Cluster weight (g)':cw,
                'Shoot number': sn,'Vine canopy (%)': vc, 'Leaf area / metre':la,'Berry_weight_g':bw}
        # row2 = [source[0], "Yield per metre", source[2], "Information", cn, cw, sn, vc, la, bw]
        # row3 = [source[0], "Yield per square metre",source[3], "Information", cn, cw, sn, vc, la, bw]
        #Arr_Quality_yield.loc[len(Arr_Quality_yield)] = row1 
        #Arr_Quality_yield.loc[len(Arr_Quality_yield)] = row2 
        #Arr_Quality_yield.loc[len(Arr_Quality_yield)] = row3 
        Quality_yield.append(row1)
        Quality_yield.append(row2)
        Quality_yield.append(row3)
    #print(Quality_yield)
    return jsonify(Quality_yield)
    #return {"row1": row1, "row2":row2, "row3": row3}
    #return jsonify{"row1": Quality_yield1, "row2":Quality_yield2, "row3": Quality_yield3}
    
def Modules(prediction_proba1, prediction_proba2):
    #module1====================================================================================================================
    Yield_per_wine = np.exp(prediction_proba1[0]-2)*10
    Yield_per_m = np.exp(prediction_proba1[1]-2)*10
    Yield_per_m2 = np.exp(prediction_proba1[2]-2)*10
    #module2====================================================================================================================
    Berry_OD280 = np.exp(prediction_proba2[0])-1
    Berry_OD320 = np.exp(prediction_proba2[1])-1
    Berry_OD520 = prediction_proba2[2]
    Juice_total_soluble_solids = np.exp(prediction_proba2[3])
    Juice_pH = np.exp(prediction_proba2[4])
    Juice_primary_amino_acids = np.exp(prediction_proba2[5])*100
    Juice_malic_acid = np.exp(prediction_proba2[6]-1)*10
    Juice_tartaric_acid = np.exp(prediction_proba2[7])
    Juice_calcium = np.exp(prediction_proba2[8])*50
    Juice_potassium = np.exp(prediction_proba2[9]+6)
    Juice_alanine = np.exp(prediction_proba2[10]-2)*100
    Juice_arginine = np.exp(prediction_proba2[11]-2)*1000
    Juice_aspartic_acid = np.exp(prediction_proba2[12]-2)*100
    Juice_serine = np.exp(prediction_proba2[13])

    source2 = [Berry_OD280,Berry_OD320,Berry_OD520,Juice_total_soluble_solids,Juice_pH,Juice_primary_amino_acids,Juice_malic_acid
                    ,Juice_tartaric_acid,Juice_calcium,Juice_potassium,Juice_alanine,Juice_arginine,Juice_aspartic_acid,Juice_serine]
    #module3 ==================================================================================================================
    Berry_OD280 = np.log(Berry_OD280/10+1)
    Berry_OD320 = np.log(Berry_OD320+1)
    Berry_OD520 = np.log(Berry_OD520+2)
    Juice_total_soluble_solids = np.log(Juice_total_soluble_solids/10)
    Juice_pH = np.log(Juice_pH)
    Juice_primary_amino_acids = np.log(Juice_primary_amino_acids/100)
    Juice_malic_acid = np.log(Juice_malic_acid+1)

    Juice_tartaric_acid = np.log(Juice_tartaric_acid)
    Juice_calcium = np.log(Juice_calcium/100+1)
    Juice_potassium = np.log(Juice_potassium/1000+1)
    Juice_alanine = np.log(Juice_alanine/1000+1)
    Juice_arginine = np.log(Juice_arginine/1000+2)
    Juice_aspartic_acid = np.log(Juice_aspartic_acid/100+3)
    Juice_serine = np.log(Juice_serine/200+2)

    ThirdMol_value = model_3.predict([[Berry_OD280, Berry_OD320, Berry_OD520, Juice_total_soluble_solids, Juice_pH, Juice_primary_amino_acids, 
           Juice_malic_acid, Juice_tartaric_acid, Juice_calcium, Juice_potassium, Juice_alanine, Juice_arginine, 
           Juice_aspartic_acid, Juice_serine]])

    Wine_alcohol = np.exp(ThirdMol_value[0][0])
    Wine_pH = np.exp(ThirdMol_value[0][1])
    Wine_monomeric_anthocyanins = np.exp(ThirdMol_value[0][2]*4)
    Wine_total_anthocyanin = (np.exp(ThirdMol_value[0][3])-1)*500
    Wine_total_phenolics = (np.exp(ThirdMol_value[0][4])-1)*20


    source3 = [Wine_alcohol,Wine_pH,Wine_monomeric_anthocyanins,Wine_total_anthocyanin,Wine_total_phenolics]
    #module4 ==================================================================================================================
    Wine_alcohol = np.log(Wine_alcohol/10)
    Wine_pH = np.log(Wine_pH)
    Wine_monomeric_anthocyanins = np.log(Wine_monomeric_anthocyanins/100)
    Wine_total_anthocyanin = (np.log(Wine_total_anthocyanin)/100)
    Wine_total_phenolics = (np.log(Wine_total_phenolics)/10)

    FourthMol_value = model_4.predict([[Wine_alcohol, Wine_pH, Wine_monomeric_anthocyanins, Wine_total_anthocyanin, Wine_total_phenolics]])
    quality = np.exp(FourthMol_value[0])+1
    quality = round(quality,2)

    Quality_yieldperwine = [quality, Yield_per_wine, Yield_per_m, Yield_per_m2,source2,source3]
    return Quality_yieldperwine

@app.route("/getModel", methods=['GET'])
def index():
    Cluster_number = 23.0
    Cluster_weight = 144.0
    Shoot_number_gt_5mm = 12.0
    Vine_canopy = 0.2
    Leaf_Area_per_m = 12000.0
    Berry_weight = 1.78

    FirstMol_value = model_1.predict([[Cluster_number, Cluster_weight, Shoot_number_gt_5mm,Berry_weight]])
    SecondMol_value = model_2.predict([[Cluster_number, Cluster_weight, Shoot_number_gt_5mm,Vine_canopy,Leaf_Area_per_m,Berry_weight]])

    firstModel = []
    seondModel = []
    
    for first in FirstMol_value:
        tmp = {"data1":first[0],"data2":first[1], "data3":first[2]}
        firstModel.append(tmp)
    print(FirstMol_value)
    return jsonify(firstModel)
    #return {"ML": firstModel}
    #return (firstModel)



if __name__ == "__main__":
    app.run()
