"""Juego del ahorcado."""

def check_palabra():
    """
    Función check_palabra().
    
    Determina la validez de la palabra ingresada. 
    Repite hasta obtener una respuesta válida.
    """
    while True:
        
        palabra = str(input('Jugador #1: ingrese una palabra: ').lower())
        
        if palabra.isalpha():
            return palabra
        else:            
            print('Incorrecto!: Ingresar solamente una palabra compuesta por letras del alfabeto español.')
            

def juego_ahorcado(palabra):
    """
    Funcion juego_ahorcado(palabra).
        
    Emplea una palabra elegida por el jugador 1, y determina si el jugador 2
    la adivina o no. Devuelve el resultado del juego.
    """
    len_palabra = len(palabra)
    palabra_j1 = list(palabra)
    for i in range(len_palabra):
        palabra_j1[i] += ' '

    print('Numero de letras de la palabra a adivinar: ' + str(len_palabra))
    palabra_j2 = ['_ ']*len_palabra
    cont = 0
    max_cont = 15
    while True:
        cont = cont + 1
        print('Palabra a adivinar: ' + ''.join(palabra_j2))
        letra = input('Jugador #2: ingrese una letra (intento #' + str(cont) + '): ')
        letra += ' '
        if letra in palabra_j1:
            idx = find_repeated_idx(palabra_j1, letra)
            print('Adivinó la letra ' + letra + '! Cantidad de ocurrencias: ' + str(idx))
            for i in idx:
                palabra_j2[i] = letra
            
            if palabra_j2==palabra_j1:
                print('Jugador #2 ganó! La palabra era: ' + palabra.upper() + '. Juego terminado.')
                return
        else:
            print('La letra ' + letra + ' no está en la palabra!')    

        if cont==max_cont:
            print('Jugador #2 perdió! Llegó al máximo número de letras sin adivinar la palabra!')
            return

def find_repeated_idx(palabra, letra):
    """
    Funcion find_repeated_idx(palabra, letra).
    
    Encuentra los indices en 'palabra' que corresponden a 'letra', inclusive si
    existen varias ocurrencias de 'letra' en 'palabra'.
    """
    idx = [i for i, value in enumerate(palabra) if value == letra]
    return idx


palabra = check_palabra()
juego_ahorcado(palabra)