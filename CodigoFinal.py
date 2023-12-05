import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.optimize import curve_fit


def lorentziana(x, x_0, gamma, delta):
    return gamma/(np.pi*((x - x_0)**2 + gamma**2)) + delta

def graficas_muestra(df, labels):
    frec = df[7,4:]
    r = df[14,4:]*rms
    x = df[20,4:]*rms
    y = df[23,4:]*rms
    fase = df[11,4:]

    fig = plt.figure()
    plt.title("Amplitud vs Frecuencia del {}, Vin = {}V".format(labels[0], labels[1]))
    plt.plot(frec/1E6, r/1E-3, c="black", linewidth=0.85)
    plt.xlabel("$f$ (MHz)")
    plt.ylabel("$r$ (mV)")
    plt.show()
    fig.savefig("Grafica_{}_{}V_r_vs_f.png".format(*labels))
    plt.close(fig)

    fig = plt.figure()
    plt.title("Fase vs Frecuencia del {}, Vin = {}V".format(labels[0], labels[1]))
    plt.plot(frec/1E6, fase, c="black", linewidth=0.85)
    plt.xlabel("$f$ (MHz)")
    plt.ylabel("$\phi$ (rad)")
    plt.show()
    fig.savefig("Grafica_{}_{}V_phi_vs_f.png".format(*labels))
    plt.close(fig)

    fig = plt.figure()
    plt.title("x vs y del {}, Vin = {}V".format(labels[0], labels[1]))
    plt.plot(x/1E-3, y/1E-3, c="black", linewidth=0.85)
    plt.axis('equal')
    plt.xlabel("$x$ (mV)")
    plt.ylabel("$y$ (mV)")
    plt.show()
    fig.savefig("Grafica_{}_{}V_y_vs_x.png".format(*labels))
    plt.close(fig)

    return None

def calculo_amplitud(df, fs_resonancia, BWs_aprox, labels):
    frec = df[7,4:]
    r = df[14,4:]*rms
    
    param = []
    stds = []

    for i in range(len(fs_resonancia)):
        ii = (np.abs(frec - fs_resonancia[i]) < BWs_aprox[i])
        
        popt, pcov = curve_fit(lorentziana, frec[ii], r[ii], p0=[fs_resonancia[i], BWs_aprox[i], np.mean(r[ii])])
        
        std = np.sqrt(np.diag(pcov))

        param.append(popt)
        stds.append(std)

        fig = plt.figure()

        label_ajuste = "$r = \gamma/{\pi[(f - f_0)^2 + \gamma^2]} + \delta$\n"
        label_ajuste += "$f_0$ = ({:.4f} $\pm$ {:.4f}) MHz\n".format(popt[0]/1E6, std[0]/1E6)
        label_ajuste += "$\gamma$ = ({:.4f} $\pm$ {:.4f}) MHz\n".format(popt[1]/1E6, std[1]/1E6)
        label_ajuste += "$\delta$ = ({:.2f} $\pm$ {:.2f}) $\mu$V".format(popt[2]/1E-6, std[2]/1E-6)

        plt.plot(frec[ii]/1E6, lorentziana(frec[ii], *popt)/1E-3, c="blue", label=label_ajuste)
        plt.plot(frec[ii]/1E6, r[ii]/1E-3, c="black", linewidth=0.85)
        plt.title("Amplitud vs Frecuencia del {}, Vin = {}V".format(labels[0], labels[1]))
        plt.xlabel("$f$ (MHz)")
        plt.ylabel("$r$ (mV)")
        plt.legend()
        plt.show()
        fig.savefig("Grafica_{}_{}V_r_vs_f_ajuste_resonancia_{}.png".format(*labels, i+1))
        plt.close(fig)

        print("Ajuste {} a {}V, R^2 = {}".format(*labels, sklearn.metrics.r2_score(r[ii], lorentziana(frec[ii], *popt))))

    return param, stds

def dfs_histeresis(indices):
    dfs = []
    long = int(len(indices)/2) + 1

    for i in range(long):
        df = pd.read_csv("V_Histeresis_normal/Datos/WTe2_Amplitud_{}V_Barrido_0.1-5MHz.csv".format(indices[i]), sep=";")
        df = np.array(df)
        dfs.append(df)

    for i in range(long, len(indices)):
        df = pd.read_csv("V_Histeresis_inversa/Datos/WTe2_Amplitud_{}V_Barrido_0.1-5MHz.csv".format(indices[i]), sep=";")
        df = np.array(df)
        dfs.append(df)

    return dfs

def graficas_datos_histeresis(dfs, indices):
    for i in range(len(dfs)):
        df = dfs[i]
        labels = ["WTe2", float(indices[i])]
        graficas_muestra(df, labels)

def histeresis(dfs, indices, fs_resonancia, BWs_aprox):
    long = len(indices)
    num_f_res = len(fs_resonancia)

    voltajes = np.array(indices, dtype=float)
    amplitudes = np.zeros([num_f_res, long])
    std_amplitudes = np.zeros([num_f_res, long])

    frecs_resonancia = np.zeros([num_f_res, long])
    std_frecs_resonancia = np.zeros([num_f_res, long])

    for i in range(long):
        df = dfs[i]
    
        labels = ["WTe2", float(indices[i])]
        param, stds = calculo_amplitud(df, fs_resonancia, BWs_aprox, labels)

        for j in range(num_f_res):
            amplitudes[j, i] = 1/(np.pi*param[j][1])
            std_amplitudes[j, i] = stds[j][1]/(np.pi*param[j][1]**2)

            frecs_resonancia[j, i] = param[j][0]
            std_frecs_resonancia[j, i] = stds[j][0]

    fig = plt.figure()
    for j in range(num_f_res):
        ii = int(long/2)
        v_normal = voltajes[0:ii + 1]
        v_inverso = voltajes[ii:]

        amp_normal = amplitudes[j][0:ii + 1]
        amp_inverso = amplitudes[j][ii:]

        std_normal = std_amplitudes[j][0:ii + 1]
        std_inverso = std_amplitudes[j][ii:]

        l_normal = frecs_resonancia[j][0:ii + 1]
        l_inverso = frecs_resonancia[j][ii:]

        std_l_normal = std_frecs_resonancia[j][0:ii + 1]
        std_l_inverso = std_frecs_resonancia[j][ii:]

        plt.scatter(v_normal, amp_normal/1E-6, label="Aumento voltaje", c="blue")
        plt.plot(v_normal, amp_normal/1E-6, c="blue")
        plt.errorbar(v_normal, amp_normal/1E-6, yerr=std_normal/1E-6, fmt='.', c="blue")


        #plt.arrow((v_normal[0:ii] + v_normal[1:])/2, (amp_normal[0:ii] + amp_normal[1:])/2, 0.01, 0.01, shape='full', lw=0, length_includes_head=True, head_width=.05)

        plt.scatter(v_inverso, amp_inverso/1E-6, label="Disminución voltaje", c="red")
        plt.plot(v_inverso, amp_inverso/1E-6, c="red")
        plt.errorbar(v_inverso, amp_inverso/1E-6, yerr=std_inverso/1E-6, fmt='.', c="red")

    
    plt.title("Amplitud picos vs Voltaje del WTe2")
    plt.xlabel("$V$ (V)")
    plt.ylabel("$Amplitud$ ($\mu$V)")
    plt.legend()
    plt.show()
    plt.close(fig)
    
    return None

#Constantes
rms = np.sqrt(2)

#Si

df = pd.read_csv("RUS_Si/Datos/RUS_Si_1V.csv", sep=";")
df = np.array(df)
labels = ["Si", "1"]
graficas_muestra(df, labels)

#WTe2
#Amplitud barrido denso

df = pd.read_csv("Barrido_denso/Datos/WTe2_Amplitud_1.000V_Barrido_0.1-2MHz.csv", sep=";")
df = np.array(df)
labels = ["WTe2", "1"]

fs_resonancia = [0.79E6]
BWs_aprox = [0.04E6]

graficas_muestra(df, labels)
param, stds = calculo_amplitud(df, fs_resonancia, BWs_aprox, labels)

#Amplitud Histéresis

indices = ["0.250", "0.500", "0.750", "1.000", "1.250", "1.500"]
indices += indices[0:len(indices) - 1][::-1]

dfs = dfs_histeresis(indices)
graficas_datos_histeresis(dfs, indices)

fs_resonancia = [0.7901E6]
BWs_aprox = [0.025E6]
histeresis(dfs, indices, fs_resonancia, BWs_aprox)
