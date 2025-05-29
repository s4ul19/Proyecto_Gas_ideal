import random
import math
import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.optimize import curve_fit
import yaml

with open("config.yaml", "r") as archivo:
    config = yaml.safe_load(archivo)
N= config["parametros"]["N"]
Lx= config["parametros"]["Lx"]
Ly= config["parametros"]["Ly"]
dt= config["parametros"]["dt"]
k_B=config["parametros"]["k_B"]
desv=config["parametros"]["desv"]
pasos=config["parametros"]["pasos"]
radio=config["parametros"]["radio"]
m=config["parametros"]["m"]

#Clase que genera las particulas 
class Particula:
    def __init__(self):
        self.vx=np.random.normal(0,desv)
        self.vy=np.random.normal(0,desv)
        self.x0=random.uniform(0.5,Lx-0.5)
        self.y0=random.uniform(0.5,Ly-0.5)       
#lista inicial
listaPosicion=[]
listaMomento=[]
def generador(listaPosicion, listaMomento):
    for i in range(N):
        A=Particula()
        B=[A.x0,A.y0]
        C=[A.vx,A.vy]
        listaPosicion.append(B)
        listaMomento.append(C)
def graficarPosicion(lista1,lista2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Primer gráfico
    axs[0].scatter(Parte_x(lista1), Parte_y(lista1), color='blue')
    axs[0].set_title("Posiciones iniciales ")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True)

    # Segundo gráfico
    axs[1].scatter(Parte_x(lista2), Parte_y(lista2), color='red')
    axs[1].set_title("Posiciones finales")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].grid(True)

    plt.tight_layout()
    #plt.show()  
def graficarHisto(Lista1,Lista2,Lista3):     # Esta funcion grafica 3 histogramas, Orden escogido: Magnitudes, vel_x, vel_y
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Histograma de las magnitudes de las velocidades
    axs[0].hist(Lista1, bins=30, density=True, alpha=0.6, color='g')
    axs[0].set_title('Distribución de la magnitud de las velocidades')
    axs[0].set_xlabel('Velocidad')
    axs[0].set_ylabel('Densidad')

    # Histograma de la componente x de las velocidades
    axs[1].hist(Lista2, bins=30, density=True, alpha=0.6, color='b')
    axs[1].set_title('Distribución de la componente x de las velocidades')
    axs[1].set_xlabel('Velocidad en x')
    axs[1].set_ylabel('Densidad')

    # Histograma de la componente y de las velocidades
    axs[2].hist(Lista3, bins=30, density=True, alpha=0.6, color='r')
    axs[2].set_title('Distribución de la componente y de las velocidades')
    axs[2].set_xlabel('Velocidad en y')
    axs[2].set_ylabel('Densidad')

    plt.tight_layout()
    #plt.show()


#Este metodo comentado era la previa forma de simular.
"""
def Simulacion(listaPosicion, listaMomento,pasos):  # De argumento van los numeros de pasos que se deseen calcular.
    for t in range(pasos):
    # Actualizar posiciones
        for i in range(N):
            listaPosicion[i][0] += float(listaMomento[i][0]) * dt
            listaPosicion[i][1] += float(listaMomento[i][1]) * dt

        # Colisiones con paredes
        for i in range(N):
            if listaPosicion[i][0] >= Lx - radio or listaPosicion[i][0] <= radio:
                listaMomento[i][0] *= -1
            if listaPosicion[i][1] >= Ly - radio or listaPosicion[i][1] <= radio:
                listaMomento[i][1] *= -1

        
        # Colisiones entre partículas
        for i in range(N):
            for j in range(i+1, N):
                dx = listaPosicion[j][0] - listaPosicion[i][0]
                dy = listaPosicion[j][1] - listaPosicion[i][1]
                distancia = math.sqrt(dx**2 + dy**2)

                if distancia < 2 * radio:  # colisión
                    # Posiciones y velocidades
                    xi, yi = listaPosicion[i]
                    xj, yj = listaPosicion[j]
                    vxi, vyi = listaMomento[i]
                    vxj, vyj = listaMomento[j]

                # Vector relativo de posición y velocidad
                    rx = xi - xj
                    ry = yi - yj
                    rvx = vxi - vxj
                    rvy = vyi - vyj

                # Producto escalar y norma al cuadrado
                    prod_escalar = rvx * rx + rvy * ry
                    dist2 = rx**2 + ry**2

                    if dist2 == 0:  # Para evitar división por cero
                        continue

                    factor = prod_escalar / dist2

                # Actualizamos velocidades (asumiendo masa igual)
                    listaMomento[i][0] -= factor * rx
                    listaMomento[i][1] -= factor * ry
                    listaMomento[j][0] += factor * rx
                    listaMomento[j][1] += factor * ry
"""
def obtener_celda(x, y, cell_size):
    return int(x // cell_size), int(y // cell_size)

def Simulacion(listaPosicion, listaMomento, pasos):
    cell_size = 2 * radio
    grid_cols = int(Lx // cell_size)
    grid_rows = int(Ly // cell_size)

    for t in range(pasos):
        # Actualizar posiciones
        for i in range(N):
            listaPosicion[i][0] += listaMomento[i][0] * dt
            listaPosicion[i][1] += listaMomento[i][1] * dt

        # Colisiones con paredes
        for i in range(N):
            if listaPosicion[i][0] >= Lx - radio or listaPosicion[i][0] <= radio:
                listaMomento[i][0] *= -1
            if listaPosicion[i][1] >= Ly - radio or listaPosicion[i][1] <= radio:
                listaMomento[i][1] *= -1

        # Construir rejilla
        grid = {}
        for i in range(N):
            cx, cy = obtener_celda(listaPosicion[i][0], listaPosicion[i][1], cell_size)
            if (cx, cy) not in grid:
                grid[(cx, cy)] = []
            grid[(cx, cy)].append(i)

        # Verificar colisiones locales
        for (cx, cy), indices in grid.items():
            # Ver celdas vecinas (3x3 alrededor)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) not in grid:
                        continue
                    vecinos = grid[(nx, ny)]
                    for i in indices:
                        for j in vecinos:
                            if j <= i:
                                continue
                            dx = listaPosicion[j][0] - listaPosicion[i][0]
                            dy = listaPosicion[j][1] - listaPosicion[i][1]
                            distancia = math.sqrt(dx**2 + dy**2)
                            if distancia < 2 * radio:
                                # Mismo tratamiento de colisión que antes
                                xi, yi = listaPosicion[i]
                                xj, yj = listaPosicion[j]
                                vxi, vyi = listaMomento[i]
                                vxj, vyj = listaMomento[j]

                                rx = xi - xj
                                ry = yi - yj
                                rvx = vxi - vxj
                                rvy = vyi - vyj

                                dist2 = rx**2 + ry**2
                                if dist2 == 0:
                                    continue

                                prod_escalar = rvx * rx + rvy * ry
                                factor = prod_escalar / dist2

                                listaMomento[i][0] -= factor * rx
                                listaMomento[i][1] -= factor * ry
                                listaMomento[j][0] += factor * rx
                                listaMomento[j][1] += factor * ry

# Estas dos funciones extraen la información de las coordenadas de las listas.
def Parte_x(lista):
    CoordenadasX= []
    for t in range(len(lista)):
        CoordenadasX.append(lista[t][0])
    return CoordenadasX

def Parte_y(lista):
    CoordenadasY= []
    for t in range(len(lista)):
        CoordenadasY.append(lista[t][1])
    return CoordenadasY

def Magnitud(lista1,lista2):
    listaMagnitud=[]
    for t in range(len(lista1)):
        M= math.sqrt((lista1[t])**2+(lista2[t]**2))
        listaMagnitud.append(M)
    return listaMagnitud

def Cinetica_med(lista):
    Cinetica=0
    for t in range(len(lista)):
        Cinetica += 0.5*m*(lista[t])**2
    return Cinetica/N
def Temperatura(cinetica):
    return (cinetica)/(k_B)
def vel_med(lista):
    vel_med=0
    for t in range(len(lista)):
        vel_med+=lista[t]
    return vel_med/N
# APLICACION DEL CODIGO
generador(listaPosicion,listaMomento)

listaPosicion_inicial=copy.deepcopy(listaPosicion)
listaMomento_inicial=copy.deepcopy(listaMomento)

Simulacion(listaPosicion,listaMomento,pasos)

listaMomento_x=Parte_x(listaMomento)
listaMomento_y=Parte_y(listaMomento)
Magnitud_Momento=Magnitud(listaMomento_x,listaMomento_y)

graficarHisto(Magnitud_Momento,listaMomento_x,listaMomento_y)
graficarPosicion(listaPosicion_inicial,listaPosicion)


#<<<<<<<<<<<<<<<<<<<<ANALISIS DE DATOS<<<<<<<<<<<<<<<<<<<<<<
Kinetic=Cinetica_med(Magnitud_Momento)
T=Temperatura(Kinetic)

v_magnitudes = np.array(Magnitud_Momento)  # Lista de magnitudes de velocidad

# Función de Maxwell-Boltzmann en 2D
def maxwell_boltzmann_2d(v, T):
    factor = (m / (k_B * T))
    return factor * v * np.exp(-0.5 * factor * v**2)

v_mp= math.sqrt((k_B*T)/m)    #Velocidad esperada teoricamente
v_med=math.sqrt((math.pi*k_B*T)/(2*m))





# Ajuste de la curva a los datos del histograma
hist, bin_edges = np.histogram(v_magnitudes, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
params, covariance = curve_fit(maxwell_boltzmann_2d, bin_centers, hist, p0=[T])
T_fit = params[0]

# Graficar histograma y ajuste
plt.figure(figsize=(10, 6))
plt.hist(v_magnitudes, bins=30, density=True, alpha=0.6, color='g', label='Datos simulados')
v_range = np.linspace(0, 3 * np.sqrt(2 * k_B * T / m), 300)
plt.plot(v_range, maxwell_boltzmann_2d(v_range, T_fit), 'r-', lw=2, 
         label=f'Maxwell-Boltzmann (T = {T_fit:.1f} K)')
plt.xlabel('Magnitud de la velocidad (m/s)')
plt.ylabel('Densidad de probabilidad')
plt.title('Ajuste de la distribución de Maxwell-Boltzmann')
plt.legend()
plt.grid(True)
plt.show()




print(f"Las moleculas probadas tenian un radio promedio de {radio}m con una masa fija de {m}kg\n\nLa energia cinetica media por molecula es {Kinetic}J\n\n La temperatura medida sería de {T}K\n\n")
print(f"Velocidad esperada segun la teoria es {v_mp} m/s \n Velocidad esperada segun la simulacion {desv} m/s\n\n Velocidad media segun la teoria {v_med} m/s\n Velocidad media segun la simulacion {vel_med(Magnitud_Momento)} m/s\n\n Se simularon {N} particulas ")