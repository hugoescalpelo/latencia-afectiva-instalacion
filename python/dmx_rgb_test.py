#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ola.ClientWrapper import ClientWrapper
import time
import array

UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56    # tu rango
FPS = 44                      # frecuencia de envío
DELAY = 1.0                   # segundos entre colores

# Buffer DMX como array('B') (tiene .tobytes() que usa ola-python internamente)
dmx = array.array('B', [0] * 512)

def set_rgb(r, g, b):
    """Escribe (r,g,b) secuencial en el rango DMX_LOW..DMX_HIGH."""
    for i in range(DMX_LOW - 1, DMX_HIGH, 3):
        if i + 2 < len(dmx):
            dmx[i]     = r
            dmx[i + 1] = g
            dmx[i + 2] = b

def main():
    wrapper = ClientWrapper()
    client = wrapper.Client()

    colors = [
        (255,   0,   0),  # rojo
        (  0, 255,   0),  # verde
        (  0,   0, 255),  # azul
        (255, 255, 255),  # blanco
        (  0,   0,   0),  # off
    ]
    idx = 0
    next_switch = time.time()

    print("Iniciando prueba RGB continua en Universe 0 (canales 22–56)...")
    try:
        while True:
            now = time.time()
            if now >= next_switch:
                r, g, b = colors[idx]
                set_rgb(r, g, b)
                print(f"DMX → R:{r} G:{g} B:{b}")
                idx = (idx + 1) % len(colors)
                next_switch = now + DELAY

            # ENVÍO CONTINUO (DMX stream)
            client.SendDmx(UNIVERSE, dmx, lambda s: None)
            wrapper.RunOnce()        # procesa IO de OLA
            time.sleep(1.0 / FPS)    # regula FPS

    except KeyboardInterrupt:
        # Apaga todo al salir
        set_rgb(0, 0, 0)
        client.SendDmx(UNIVERSE, dmx, lambda s: None)
        wrapper.RunOnce()
        print("\nPrueba detenida. Canales restaurados a 0.")

if __name__ == "__main__":
    main()
