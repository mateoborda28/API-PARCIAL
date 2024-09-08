import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
import pickle
from scipy import stats
from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
from pycaret.classification import get_config
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")


import pickle
# Cargar el modelo entrenado de preferencia
with open('third_best_model.pkl','rb') as file:
    modelo = pickle.load(file)

df = pd.read_csv('train.csv')
prueba = pd.read_csv("test.csv")
df.columns = df.columns.str.replace("'", "")
prueba.columns = prueba.columns.str.replace("'", "")


st.title("Predicción del estado de un estudiante")

# Capturar entradas del usuario y convertir a float donde sea necesario
id_input = int(st.text_input("Ingrese id:", value="0"))
marital_status_input = int(st.text_input("Ingrese Marital status:", value="1"))
application_mode_input = int(st.text_input("Ingrese Application mode:", value="1"))
application_order_input = int(st.text_input("Ingrese Application order:", value="1"))
course_input = int(st.text_input("Ingrese Course:", value="9238"))
daytime_evening_attendance_input = int(st.text_input("Ingrese Daytime/evening attendance:", value="1"))
previous_qualification_input = int(st.text_input("Ingrese Previous qualification:", value="1"))
nacionality_input = int(st.text_input("Ingrese Nacionality:", value="1"))
mothers_qualification_input = int(st.text_input("Ingrese Mother's qualification:", value="1"))
fathers_qualification_input = int(st.text_input("Ingrese Father's qualification:", value="19"))
mothers_occupation_input = int(st.text_input("Ingrese Mother's occupation:", value="5"))
fathers_occupation_input = int(st.text_input("Ingrese Father's occupation:", value="5"))
displaced_input = int(st.text_input("Ingrese Displaced:", value="0"))
educational_special_needs_input = int(st.text_input("Ingrese Educational special needs:", value="0"))
debtor_input = int(st.text_input("Ingrese Debtor:", value="0"))
tuition_fees_up_to_date_input = int(st.text_input("Ingrese Tuition fees up to date:", value="1"))
gender_input = int(st.text_input("Ingrese Gender:", value="0"))
scholarship_holder_input = int(st.text_input("Ingrese Scholarship holder:", value="1"))
international_input = int(st.text_input("Ingrese International:", value="0"))

# Las entradas del selectbox no necesitan cambios ya que son numéricas
previous_qualification_grade_input = float(st.selectbox("Seleccione Previous qualification (grade):", ['126.0']))
admission_grade_input = float(st.selectbox("Seleccione Admission grade:", ['122.6']))
age_at_enrollment_input = int(st.selectbox("Seleccione Age at enrollment:", ['18']))
curricular_units_1st_sem_credited_input = int(st.selectbox("Seleccione Curricular units 1st sem (credited):", ['0']))
curricular_units_1st_sem_enrolled_input = int(st.selectbox("Seleccione Curricular units 1st sem (enrolled):", ['6']))
curricular_units_1st_sem_evaluations_input = int(st.selectbox("Seleccione Curricular units 1st sem (evaluations):", ['6']))
curricular_units_1st_sem_approved_input = int(st.selectbox("Seleccione Curricular units 1st sem (approved):", ['6']))
curricular_units_1st_sem_grade_input = float(st.selectbox("Seleccione Curricular units 1st sem (grade):", ['14.5']))
curricular_units_1st_sem_without_evaluations_input = int(st.selectbox("Seleccione Curricular units 1st sem (without evaluations):", ['0']))
curricular_units_2nd_sem_credited_input = int(st.selectbox("Seleccione Curricular units 2nd sem (credited):", ['0']))
curricular_units_2nd_sem_enrolled_input = int(st.selectbox("Seleccione Curricular units 2nd sem (enrolled):", ['6']))
curricular_units_2nd_sem_evaluations_input = int(st.selectbox("Seleccione Curricular units 2nd sem (evaluations):", ['7']))
curricular_units_2nd_sem_approved_input = int(st.selectbox("Seleccione Curricular units 2nd sem (approved):", ['6']))
curricular_units_2nd_sem_grade_input = float(st.selectbox("Seleccione Curricular units 2nd sem (grade):", ['12.428571428571429']))
curricular_units_2nd_sem_without_evaluations_input = int(st.selectbox("Seleccione Curricular units 2nd sem (without evaluations):", ['0']))
unemployment_rate_input = float(st.selectbox("Seleccione Unemployment rate:", ['11.1']))
inflation_rate_input = float(st.selectbox("Seleccione Inflation rate:", ['0.6']))
gdp_input = float(st.selectbox("Seleccione GDP:", ['2.02']))



nueva_fila = pd.DataFrame({
    'id': [id_input],
    'Marital status': [marital_status_input],
    'Application mode': [application_mode_input],
    'Application order': [application_order_input],
    'Course': [course_input],
    'Daytime/evening attendance': [daytime_evening_attendance_input],
    'Previous qualification': [previous_qualification_input],
    'Previous qualification (grade)': [previous_qualification_grade_input],
    'Nacionality': [nacionality_input],
    "Mothers qualification": [mothers_qualification_input],
    "Fathers qualification": [fathers_qualification_input],
    "Mothers occupation": [mothers_occupation_input],
    "Fathers occupation": [fathers_occupation_input],
    'Admission grade': [admission_grade_input],
    'Displaced': [displaced_input],
    'Educational special needs': [educational_special_needs_input],
    'Debtor': [debtor_input],
    'Tuition fees up to date': [tuition_fees_up_to_date_input],
    'Gender': [gender_input],
    'Scholarship holder': [scholarship_holder_input],
    'Age at enrollment': [age_at_enrollment_input],
    'International': [international_input],
    'Curricular units 1st sem (credited)': [curricular_units_1st_sem_credited_input],
    'Curricular units 1st sem (enrolled)': [curricular_units_1st_sem_enrolled_input],
    'Curricular units 1st sem (evaluations)': [curricular_units_1st_sem_evaluations_input],
    'Curricular units 1st sem (approved)': [curricular_units_1st_sem_approved_input],
    'Curricular units 1st sem (grade)': [curricular_units_1st_sem_grade_input],
    'Curricular units 1st sem (without evaluations)': [curricular_units_1st_sem_without_evaluations_input],
    'Curricular units 2nd sem (credited)': [curricular_units_2nd_sem_credited_input],
    'Curricular units 2nd sem (enrolled)': [curricular_units_2nd_sem_enrolled_input],
    'Curricular units 2nd sem (evaluations)': [curricular_units_2nd_sem_evaluations_input],
    'Curricular units 2nd sem (approved)': [curricular_units_2nd_sem_approved_input],
    'Curricular units 2nd sem (grade)': [curricular_units_2nd_sem_grade_input],
    'Curricular units 2nd sem (without evaluations)': [curricular_units_2nd_sem_without_evaluations_input],
    'Unemployment rate': [unemployment_rate_input],
    'Inflation rate': [inflation_rate_input],
    'GDP': [gdp_input]
})


prueba = pd.concat([prueba, nueva_fila], ignore_index=True)
fila_5 = prueba.iloc[4]
ultima_fila = prueba.iloc[-1]

prueba.iloc[-1] = ultima_fila.astype(fila_5.dtypes)

def generar_base():
    ct = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance',
      'Previous qualification', 'Nacionality', 'Mothers qualification', 'Fathers qualification',
      'Mothers occupation', 'Fathers occupation', 'Displaced', 'Educational special needs', 'Debtor',
      'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
    

    for k in ct:
        df[k] = df[k].astype("O")
        prueba[k] = prueba[k].astype("O")

    # Faltantes
    # Train
    ft = pd.DataFrame(df.isnull().sum()).reset_index()
    ft.columns = ["Variable","Faltantes"]
    ft["% Faltantes"] = ft["Faltantes"] * 100 / df.shape[0]
    #ft.loc[ft["% Faltantes"] > 0]

    formato = pd.DataFrame({'Variable': list(df.columns), 'Formato': df.dtypes })
    ft = pd.merge(ft, formato, on=["Variable"], how="left")

    # Test
    ft2 = pd.DataFrame(prueba.isnull().sum()).reset_index()
    ft2.columns = ["Variable","Faltantes"]
    ft2["% Faltantes"] = ft2["Faltantes"] * 100 / prueba.shape[0]
    #ft2.loc[ft2["% Faltantes"] > 0]

    # Identificar variables cuantitativas
    cuantitativas = list(formato.loc[formato["Formato"] != "object", "Variable"])
    cuantitativas = [x for x in cuantitativas if x not in ["id", "Target"]]

    # Ingeniería de variables
        
    # Variables al cuadrado
    def prueba_kr(x):
        return 1 if x <= 0.10 else 0

    def criterion_(df, columns):
        for k in columns:
            df[k] = df[k].map(prueba_kr)
        df["criterio"] = np.sum(df.get(columns), axis=1)
        df["criterio"] = df.apply(lambda row: 1 if row["criterio"] == 3 else 0, axis=1)

    base_cuadrado = df.get(cuantitativas).copy()
    base_cuadrado["Target"] = df["Target"].copy()

    var_names2, pvalue1 = [], []

    for k in cuantitativas:
        base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2

        mue1 = base_cuadrado.loc[base_cuadrado["Target"] == "Graduate", k+"_2"].to_numpy()
        mue2 = base_cuadrado.loc[base_cuadrado["Target"] == "Dropout", k+"_2"].to_numpy()
        mue3 = base_cuadrado.loc[base_cuadrado["Target"] == "Enrolled", k+"_2"].to_numpy()
        p1 = stats.kruskal(mue1, mue2, mue3)[1]

        var_names2.append(k+"_2")
        pvalue1.append(np.round(p1, 2))

    pcuadrado1 = pd.DataFrame({'Variable2': var_names2, 'p value': pvalue1})
    pcuadrado1["criterio"] = pcuadrado1.apply(lambda row: 1 if row["p value"] <= 0.10 else 0, axis=1)


    # Interacciones cuantitativas
    lista_inter = list(combinations(cuantitativas, 2))
    base_interacciones = df.get(cuantitativas).copy()
    var_interaccion, pv1 = [], []
    base_interacciones["Target"] = df["Target"].copy()

    for k in lista_inter:
        base_interacciones[k[0]+"__"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]

        # Prueba de Kruskal
        mue1 = base_interacciones.loc[base_interacciones["Target"] == "Graduate", k[0]+"__"+k[1]].to_numpy()
        mue2 = base_interacciones.loc[base_interacciones["Target"] == "Dropout", k[0]+"__"+k[1]].to_numpy()
        mue3 = base_interacciones.loc[base_interacciones["Target"] == "Enrolled", k[0]+"__"+k[1]].to_numpy()

        p1 = stats.kruskal(mue1, mue2, mue3)[1]

        var_interaccion.append(k[0]+"__"+k[1])
        pv1.append(np.round(p1, 2))

    pxy = pd.DataFrame({'Variable': var_interaccion, 'p value': pv1})
    pxy["criterio"] = pxy.apply(lambda row: 1 if row["p value"] <= 0.10 else 0, axis=1)


    # Razones
    raz1 = [(x, y) for x in cuantitativas for y in cuantitativas]
    base_razones1 = df.get(cuantitativas).copy()
    base_razones1["Target"] = df["Target"].copy()

    var_nm, pval = [], []
    for j in raz1:
        if j[0] != j[1]:
            base_razones1[j[0]+"__coc__"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]] + 0.01)

            # Prueba de Kruskal
            mue1 = base_razones1.loc[base_razones1["Target"] == "Graduate", j[0] + "__coc__" + j[1]].to_numpy()
            mue2 = base_razones1.loc[base_razones1["Target"] == "Dropout", j[0] + "__coc__" + j[1]].to_numpy()
            mue3 = base_razones1.loc[base_razones1["Target"] == "Enrolled", j[0] + "__coc__" + j[1]].to_numpy()

            p1 = stats.kruskal(mue1, mue2, mue3)[1]

            # Guardar valores
            var_nm.append(j[0]+"__coc__"+j[1])
            pval.append(np.round(p1, 2))

    prazones = pd.DataFrame({'Variable': var_nm, 'p value': pval})
    prazones["criterio"] = prazones.apply(lambda row: 1 if row["p value"] <= 0.10 else 0, axis=1)

    # Interacciones categóricas
    categoricas = list(formato.loc[formato["Formato"] == "O", "Variable"])
    categoricas = [x for x in categoricas if x not in ["id", "Target"]]

    def nombre_(x):
        return "C" + str(x)

    cb = list(combinations(categoricas, 2))
    p_value, modalidades, nombre_var = [], [], []
    base2 = df.get(categoricas).copy()

    for k in base2.columns:
        base2[k] = base2[k].map(nombre_)

    base2["Target"] = df["Target"].copy()

    for k in range(len(cb)):
        # Variable con interacción
        base2[cb[k][0]] = base2[cb[k][0]]
        base2[cb[k][1]] = base2[cb[k][1]]

        base2[cb[k][0]+"__"+cb[k][1]] = base2[cb[k][0]] + "__" + base2[cb[k][1]]

        # Prueba chi cuadrado
        c1 = pd.DataFrame(pd.crosstab(base2["Target"], base2[cb[k][0]+"__"+cb[k][1]]))
        pv = stats.chi2_contingency(c1)[1]

        # Número de modalidades por categoría
        mod_ = len(base2[cb[k][0]+"__"+cb[k][1]].unique())

        # Guardar p value y modalidades
        nombre_var.append(cb[k][0]+"__"+cb[k][1])
        modalidades.append(mod_)
        p_value.append(pv)

    pc = pd.DataFrame({'Variable': nombre_var, 'Num Modalidades': modalidades, 'p value': p_value})
    #pc.loc[(pc["p value"] <= 0.20) & (pc["Num Modalidades"] <= 60), ].sort_values(["p value"], ascending=True)

    # Dummies categóricas más significativas (p value <= 0.20 y bajo número de modalidades)
    def indicadora(x):
        return 1 if x else 0

    seleccion1 = list(pc.loc[(pc["p value"] <= 0.20) & (pc["Num Modalidades"] <= 60), "Variable"])
    sel1 = base2.get(seleccion1)

    contador = 0
    for k in sel1:
        if contador == 0:
            lb1 = pd.get_dummies(sel1[k], drop_first=True)
            lb1.columns = [k + "_" + x for x in lb1.columns]
        else:
            lb2 = pd.get_dummies(sel1[k], drop_first=True)
            lb2.columns = [k + "_" + x for x in lb2.columns]
            lb1 = pd.concat([lb1, lb2], axis=1)
        contador = contador + 1

    for k in lb1.columns:
        lb1[k] = lb1[k].map(indicadora)

    lb1["Target"] = df["Target"].copy()


    # Interacción cuantitativa vs categórica
    cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]
    cat_cuanti[0]
    v1, v2, pvalores_min, pvalores_max = [], [], [], []

    for j in cat_cuanti:
        k1 = j[0]
        k2 = j[1]

        g1 = pd.get_dummies(df[k2])
        lt1 = list(g1.columns)

        for k in lt1:
            g1[k] = g1[k] * df[k1]

        g1["Target"] = df["Target"].copy()

        pvalues_c = []
        for y in lt1:
            mue1 = g1.loc[g1["Target"] == "Graduate", y].to_numpy()
            mue2 = g1.loc[g1["Target"] == "Dropout", y].to_numpy()
            mue3 = g1.loc[g1["Target"] == "Enrolled", y].to_numpy()

            try:
                pval = (stats.kruskal(mue1, mue2, mue3)[1] <= 0.20)
                pval = 1 if pval else 0
            except ValueError:
                pval = 0
            pvalues_c.append(pval)

        min_ = np.min(pvalues_c)  # Se revisa si alguna de las categorías no es significativa
        max_ = np.max(pvalues_c)  # Se revisa si alguna de las categorías es significativa
        v1.append(k1)  # nombre de la variable 1
        v2.append(k2)  # nombre de la variable 2
        pvalores_min.append(np.round(min_, 2))
        pvalores_max.append(np.round(max_, 2))

    pc2 = pd.DataFrame({'Cuantitativa': v1, 'Categórica': v2, 'p value': pvalores_min, 'p value max': pvalores_max})
    #pc2.loc[(pc2["p value"] == 1) & (pc2["p value max"] == 1), ]

    v1 = list(pc2.loc[(pc2["p value"] == 1) & (pc2["p value max"] == 1), "Cuantitativa"])
    v2 = list(pc2.loc[(pc2["p value"] == 1) & (pc2["p value max"] == 1), "Categórica"])

    for j in range(len(v1)):
        if j == 0:
            g1 = pd.get_dummies(df[v2[j]], drop_first=True)
            lt1 = list(g1.columns)
            for k in lt1:
                g1[k] = g1[k] * df[v1[j]]
            g1.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
        else:
            g2 = pd.get_dummies(df[v2[j]], drop_first=True)
            lt1 = list(g2.columns)
            for k in lt1:
                g2[k] = g2[k] * df[v1[j]]
            g2.columns = [v1[j] + "_" + v2[j] + "_" + str(x) for x in lt1]
            g1 = pd.concat([g1, g2], axis=1)

    g1["Target"] = df["Target"].copy()

    # Selección de variables con Xgboost
    var_cuad = list(pcuadrado1["Variable2"])
    base_modelo1 = base_cuadrado.get(var_cuad + ["Target"])
    mapping = {'Graduate': 0, 'Dropout': 1, 'Enrolled': 2}
    base_modelo1["Target"] = base_modelo1["Target"].map(mapping)

    cov = list(base_modelo1.columns)
    cov = [x for x in cov if x not in ["Target"]]

    X1 = base_modelo1.get(cov)
    y1 = base_modelo1.get(["Target"])

    modelo1 = XGBClassifier()
    modelo1 = modelo1.fit(X1, y1)

    importancias = modelo1.feature_importances_
    imp1 = pd.DataFrame({'Variable': X1.columns, 'Importancia': importancias})
    imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
    imp1 = imp1.sort_values(["Importancia"], ascending=False)
    imp1.index = range(imp1.shape[0])

    # Interacciones cuantitativas
    var_int = list(pxy["Variable"])
    base_modelo2 = base_interacciones.get(var_int + ["Target"])
    base_modelo2["Target"] = base_modelo2["Target"].map(mapping)

    cov = list(base_modelo2.columns)
    cov = [x for x in cov if x not in ["Target"]]

    X2 = base_modelo2.get(cov)
    y2 = base_modelo2.get(["Target"])

    modelo2 = XGBClassifier()
    modelo2 = modelo2.fit(X2, y2)

    importancias = modelo2.feature_importances_
    imp2 = pd.DataFrame({'Variable': X2.columns, 'Importancia': importancias})
    imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
    imp2 = imp2.sort_values(["Importancia"], ascending=False)
    imp2.index = range(imp2.shape[0])

    # Razones
    var_raz = list(prazones["Variable"])
    base_modelo3 = base_razones1.get(var_raz + ["Target"])
    base_modelo3["Target"] = base_modelo3["Target"].map(mapping)

    cov = list(base_modelo3.columns)
    cov = [x for x in cov if x not in ["Target"]]

    X3 = base_modelo3.get(cov)
    y3 = base_modelo3.get(["Target"])

    modelo3 = XGBClassifier()
    modelo3 = modelo3.fit(X3, y3)

    importancias = modelo3.feature_importances_
    imp3 = pd.DataFrame({'Variable': X3.columns, 'Importancia': importancias})
    imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
    imp3 = imp3.sort_values(["Importancia"], ascending=False)
    imp3.index = range(imp3.shape[0])

    # Interacciones categóricas
    lb1["Target"] = lb1["Target"].map(mapping)

    cov = list(lb1.columns)
    cov = [x for x in cov if x not in ["Target"]]

    X4 = lb1.get(cov)
    y4 = lb1.get(["Target"])

    modelo4 = XGBClassifier()
    modelo4 = modelo4.fit(X4, y4)

    importancias = modelo4.feature_importances_
    imp4 = pd.DataFrame({'Variable': X4.columns, 'Importancia': importancias})
    imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
    imp4 = imp4.sort_values(["Importancia"], ascending=False)
    imp4.index = range(imp4.shape[0])

    # Interacción cuantitativa - categórica
    g1["Target"] = g1["Target"].map(mapping)

    cov = list(g1.columns)
    cov = [x for x in cov if x not in ["Target"]]

    X5 = g1.get(cov)
    y5 = g1.get(["Target"])

    modelo5 = XGBClassifier()
    modelo5 = modelo5.fit(X5, y5)

    importancias = modelo5.feature_importances_
    imp5 = pd.DataFrame({'Variable': X5.columns, 'Importancia': importancias})
    imp5["Importancia"] = imp5["Importancia"] * 100 / np.sum(imp5["Importancia"])
    imp5 = imp5.sort_values(["Importancia"], ascending=False)
    imp5.index = range(imp5.shape[0])

    # Variables más importantes por Xgboost para cada caso
    c2 = list(imp1.iloc[0:5, 0])  # Variables al cuadrado
    cxy = list(imp2.iloc[0:5, 0])  # Interacciones cuantitativas
    razxy = list(imp3.iloc[0:5, 0])  # Razones
    catxy = list(imp4.iloc[0:5, 0])  # Interacciones categóricas
    cuactxy = list(imp5.iloc[0:5, 0])  # Interacción cuantitativa y categórica

    # Preparación de los datos
    # Variables cuantitativas (Activar D1)
    D1 = df.get(cuantitativas).copy()

    # Variables categóricas
    D2 = df.get(categoricas).copy()
    for k in categoricas:
        D2[k] = D2[k].map(nombre_)
    D4 = D2.copy()

    # Variables al cuadrado (Activar D1)
    cuadrado = [re.findall(r'(.+)_\d+', item) for item in c2]
    cuadrado = [x[0] for x in cuadrado]

    for k in cuadrado:
        D1[k+"_2"] = D1[k] ** 2

    # Interacciones cuantitativas (Activar D1)
    result = [re.findall(r'([A-Za-z\s\(\)0-9]+)', item) for item in cxy]

    for k in result:
        D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]

    # Razones
    result2 = [re.findall(r'(.+)__coc__(.+)', item) for item in razxy]
    for k in result2:
        k2 = k[0]
        D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]] + 0.01)

    # Interacciones categóricas
    result3 = [re.search(r'([^_]+__[^_]+)', item).group(1).split('__') for item in catxy]
    for k in result3:
        D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]

    # Interacción cuantitativa vs categórica
    D5 = df.copy()
    result4 = [re.search(r'(.+?)_(.+?)_1', item).groups() for item in cuactxy]
    contador = 0
    for k in result4:
        col1, col2 = k[1], k[0]  # categórica, cuantitativa
        if contador == 0:
            D51 = pd.get_dummies(D5[col1], drop_first=True)
            for j in D51.columns:
                D51[j] = D51[j] * D5[col2]
            D51.columns = [col2+"_"+col1+"_"+str(x) for x in D51.columns]
        else:
            D52 = pd.get_dummies(D5[col1], drop_first=True)
            for j in D52.columns:
                D52[j] = D52[j] * D5[col2]
            D52.columns = [col2+"_"+col1+"_"+str(x) for x in D52.columns]
            D51 = pd.concat([D51, D52], axis=1)
        contador = contador + 1

    # Base Modelo
    
    
        B1 = pd.concat([D1, D4], axis=1)
        base_modelo = pd.concat([B1, D51], axis=1)
        base_modelo["Target"] = df["Target"].copy()
        base_modelo["Target"] = base_modelo["Target"].map(mapping)
        base_modelo.head()
    

        # Variables cuantitativas (Activar D1)
        D1 = prueba.get(cuantitativas).copy()

        # Variables categóricas
        D2 = prueba.get(categoricas).copy()
        for k in categoricas:
            D2[k] = D2[k].map(nombre_)
        D4 = D2.copy()

        # Variables al cuadrado (Activar D1)
        for k in cuadrado:
            D1[k+"_2"] = D1[k] ** 2

        # Interacciones cuantitativas (Activar D1)
        for k in result:
            D1[k[0]+"__"+k[1]] = D1[k[0]] * D1[k[1]]

        # Razones
        for k in result2:
            k2 = k[0]
            D1[k2[0]+"__coc__"+k2[1]] = D1[k2[0]] / (D1[k2[1]] + 0.01)

        # Interacciones categóricas
        for k in result3:
            D4[k[0]+"__"+k[1]] = D4[k[0]] + "_" + D4[k[1]]

        # Interacción cuantitativa vs categórica
        D5 = prueba.copy()
        contador = 0
        for k in result4:
            col1, col2 = k[1], k[0]  # categórica, cuantitativa
            if contador == 0:
                D51 = pd.get_dummies(D5[col1], drop_first=True)
                for j in D51.columns:
                    D51[j] = D51[j] * D5[col2]
                D51.columns = [col2+"_"+col1+"_"+str(x) for x in D51.columns]
            else:
                D52 = pd.get_dummies(D5[col1], drop_first=True)
                for j in D52.columns:
                    D52[j] = D52[j] * D5[col2]
                D52.columns = [col2+"_"+col1+"_"+str(x) for x in D52.columns]
                D51 = pd.concat([D51, D52], axis=1)
            contador = contador + 1

    B1 = pd.concat([D1, D4], axis=1)
    base_modelo2 = pd.concat([B1, D51], axis=1)
    return base_modelo2

    




if st.button("Calcular"):
    
    bpred=generar_base()
    predictions = modelo.predict(bpred)
    result = pd.DataFrame({
    'id': prueba["id"],
    'Target': predictions['prediction_label']})
    mapping = {0: 'Graduate', 1: 'Dropout', 2: 'Enrolled'}
    result["Target"] = result["Target"].map(mapping)
    st.write(result.tail(1)) 
