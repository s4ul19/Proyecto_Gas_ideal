import pygame
import random
import math
import yaml
import numpy as np

factorSize=3*10**4
with open("config.yaml", "r") as archivo:
    config = yaml.safe_load(archivo)
N= config["parametros"]["N"]
Lx= config["parametros"]["Lx"]
Ly= config["parametros"]["Ly"]
dt= config["parametros"]["dt"]*10
k_B=config["parametros"]["k_B"]
desv=config["parametros"]["desv"]
pasos=config["parametros"]["pasos"]
radio=config["parametros"]["radio"]*factorSize  #Escalamiento
m=config["parametros"]["m"]


class Particula:
    def __init__(self):
        self.vx=np.random.normal(0,desv)
        self.vy=np.random.normal(0,desv)
        self.x0=random.uniform(0.5,Lx-0.5)
        self.y0=random.uniform(0.5,Ly-0.5)     
        

#lista inicial

listaPosicion=[]
listaMomento=[]
for i in range(N):
    A=Particula()
    B=[A.x0,A.y0]
    C=[A.vx,A.vy]
    listaPosicion.append(B)
    listaMomento.append(C)

def obtener_celda(x, y, cell_size):
    return int(x // cell_size), int(y // cell_size)

def Simulacion(listaPosicion, listaMomento):
    cell_size = 2 * radio
    grid_cols = int(Lx // cell_size)
    grid_rows = int(Ly // cell_size)

   
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






# Implementacion pygame
pygame.init()
screen = pygame.display.set_mode((Lx, Ly))
clock = pygame.time.Clock()

running = True
while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False 
    
    screen.fill((0,0,0))
    Simulacion(listaPosicion,listaMomento)
      
    for i in range(len(listaPosicion)):
        pygame.draw.circle(screen, (255,0,0),tuple(listaPosicion[i]),radio)

    pygame.display.flip()

pygame.quit()
