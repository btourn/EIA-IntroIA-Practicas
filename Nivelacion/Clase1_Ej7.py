"""Juego piedra-papel-tijera."""

def check_jugada(jugador):
    """
    Función check_jugada().
    
    Determina la validez de la jugada ingresada. 
    Repite hasta obtener una respuesta válida.
    """
    jugadas_posibles = ["piedra", "papel", "tijera"]
    
    while True:
        
        jugada = input("Jugador #" + str(jugador) + ": Elija piedra, papel o tijera: \n").lower()
        
        if jugada in jugadas_posibles:
            return jugada
        else:
            print("Incorrecto! Ingresar una jugada valida.")
        

def quien_gana(jugada1, jugada2):
    """
    Función quien_gana().
    
    Determina cual de los jugadores gana.
    """
    empates = {'piedra-piedra', 'papel-papel', 'tijera-tijera'}
    gana_jugador1 = {'piedra-tijera', 'papel-piedra', 'tijera-papel'}
    gana_jugador2 = {'piedra-papel', 'papel-tijera', 'tijera-piedra'}
    jugadas = jugada1 + '-' + jugada2
      
    if jugadas in empates:
      print('Empate!: jugar de nuevo.')
      return
      
    if jugadas in gana_jugador1:
      print('Gana jugador #1!')
      return
      
    if jugadas in gana_jugador2:
      print('Gana jugador #2!')
      return


jugada1 = check_jugada(1)
jugada2 = check_jugada(2)
quien_gana(jugada1, jugada2)
