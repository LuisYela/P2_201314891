import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import analizar

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
direccionImgUSAC = './static'
contenidoImgUSAC = os.listdir(direccionImgUSAC)

app = Flask(__name__)

app.config['UPLOAD_FOLDER']="./static"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about',)
def about():
    return render_template('about.html')

@app.route('/datos', methods=['POST'])
def crear_dato():
    return "datos recividos"

@app.route('/uploader', methods=['POST'])
def cargarImagenes():
    if request.method == "POST":
        f = request.files['archivo']
        filename= secure_filename(f.filename)
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return "archivo subido exitosamente"
    return "datos recividos"

@app.route('/analizar')
def analizarImagenes():
    direccionImgUSAC = './static'
    contenidoImgUSAC = os.listdir(direccionImgUSAC)
    listadoImagenes = []
    listadoUniversidades = []
    contadorU=0
    contadorL=0
    contadorM=0
    contadorR=0
    contadorE=0
    encontradosU = 0
    encontradosL = 0
    encontradosM = 0
    encontradosR = 0
    encontradosE = 0
    #print("antes del for")
    control=0
    for ficheroUSAC in contenidoImgUSAC:
        vuelta=[1]
        if os.path.isfile(os.path.join(direccionImgUSAC, ficheroUSAC)) and (ficheroUSAC.endswith('.jpg')):
            print('despues del filtro' +ficheroUSAC)
            control=control+1
            listadoImagenes.append(ficheroUSAC)
            #print(ficheroUSAC.lower())
            if 'usac' in ficheroUSAC.lower():
                #print("archivo usac -->" + ficheroUSAC)
                encontradosU = encontradosU+1
            elif 'landivar' in ficheroUSAC.lower():
                #print("archivo landivar -->" + ficheroUSAC)
                encontradosL= encontradosL+1
            elif 'mariano' in ficheroUSAC.lower():
                #print("archivo mariano -->" + ficheroUSAC)
                encontradosM=encontradosM+1
            elif 'marroquin' in ficheroUSAC.lower():
                #print("archivo marroquin -->" + ficheroUSAC)
                encontradosR=encontradosR+1
            else:
                #print("error en el nombre -->" + ficheroUSAC)
                encontradosE=encontradosE+1
            resultado = cv2.imread('./static/' + ficheroUSAC)
            imagen = np.array(resultado.reshape(-1,1).T)
            for elemento in imagen[0]:
                vuelta.append(elemento)
                #print(elemento)
            vuelta=np.array(vuelta)
            #print(vuelta)
            #print(vuelta.shape)
            resultU = analizar.modelU.predict(vuelta)
            resultL = analizar.modelL.predict(vuelta)
            resultM = analizar.modelM.predict(vuelta)
            resultR = analizar.modelR.predict(vuelta)
            #print(type(result[0]))
            #print(ficheroUSAC)
            #print(ficheroUSAC +'--> ')
            #print(resultU)
            #print(ficheroUSAC +'--> ')
            #print(resultL)
            #print(ficheroUSAC +'--> ')
            #print(resultM)
            #print(ficheroUSAC +'--> ')
            #print(resultR)
            if resultU[0]==1:
                listadoUniversidades.append('USAC')
                contadorU=contadorU+1
                pass
            elif resultL[0]==1:
                listadoUniversidades.append('Landivar')
                contadorL=contadorL+1
                pass
            elif resultM[0]==1:
                listadoUniversidades.append('Mariano')
                contadorM=contadorM+1
                pass
            elif resultR[0]==1:
                listadoUniversidades.append('Marroquin')
                contadorR=contadorR+1
                pass
            else:
                listadoUniversidades.append('UNDEFINED')
                contadorE=contadorE+1
                pass
    print(listadoImagenes)
    print(listadoUniversidades)
    """print('----------contadores-------')
    print(contadorU)
    print(contadorL)
    print(contadorM)
    print(contadorR)
    print(contadorE)
    print('----------encontrados-------')
    print(encontradosU)
    print(encontradosL)
    print(encontradosM)
    print(encontradosR)
    print(encontradosE)"""
    #print(control)
    if len(listadoImagenes) < 6:
        return render_template('about.html', imgs=listadoImagenes, noms=listadoUniversidades)
    else:
        porcentajeU=contadorU*100/control
        porcentajeL=contadorL*100/control
        porcentajeM=contadorM*100/control
        porcentajeR=contadorR*100/control
        porcentajeE=contadorE*100/control
        if encontradosU>0:
            porcentajeUs=contadorU*100/encontradosU
        else:
            porcentajeUs=contadorU*100/1
        if encontradosL>0:
            porcentajeLa=contadorL*100/encontradosL
        else:
            porcentajeLa=contadorL*100/1
        if encontradosM>0:
            porcentajeMa=contadorM*100/encontradosM
        else:
            porcentajeMa=contadorM*100/1
        if encontradosR>0:
            porcentajeRr=contadorR*100/encontradosR
        else:
            porcentajeRr=contadorR*100/1
        if encontradosE>0:
            porcentajeEr=contadorE*100/encontradosE
        else:
            porcentajeEr=contadorE*100/1
        return render_template('about.html', eU=porcentajeUs, eL=porcentajeLa, eM=porcentajeMa, eR=porcentajeRr, eE=porcentajeEr, pU=porcentajeU, pL=porcentajeL, pM=porcentajeM, pR=porcentajeR, pE=porcentajeE, enU=encontradosU, enL=encontradosL, enM=encontradosM, enR=encontradosR, enE=encontradosE, sU=contadorU, sL=contadorL, sM=contadorM, sR=contadorR, sE=contadorE)
    #print(vuelta.shape)
    #print(result.shape)
    print("esto nunca debe suceder")
    return "datos recividos"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
