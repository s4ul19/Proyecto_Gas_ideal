# Proyecto_Gas_ideal
El codigo fue hecho en linux, de tal forma que el uso del makefile puede traer problemas para usuarios de Windows asi como compatibilidad entre versiones de modulos. 


El codigo simula un gas en un espacio bidimensional, asumiendo los supuestos de que es un gas ideal.

IMPORTANTE:  (Para ver que hace cada parametro, ver config.yaml)
        En config.yaml estan los parametros de importancia para la simulacion.
        Tenga en consideracion que su actual configuracion aspira a ser fisicamente relevante
        En particular es importante notar que el parametro dt y radio son importantes para la relavancia fisica.
        El numero de pasos y el numero de particulas afectan directamente al tiempo computacional del programa. 
    
Metodos.py, tiene toda la estructura referenta a la informacion calculada. Fue diseñada con el proposito de hacer facil su modularizacion, por lo que si el usuario desea una mayor implementacion, no deberia de ser un problema.

            Basta con ejecutar el Metodos.py
            
    
Simulacion.py, tiene implementado un pygame de tal forma que pueda continuamente desplegar una animacion de los choques, es hace un escalamiento del radio, ya que de no hacerlo los radios, heredados del .yaml, serian pequeñisimo, factorSize es lo que permite dicho ajuste. 
            Basta con ejecutar Simulacion.py
