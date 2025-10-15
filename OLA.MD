# 🎛️ Guía de Configuración OLA + ENTTEC Open DMX + Raspberry Pi (Latencia Afectiva)

> Documento técnico de referencia para reconfigurar y depurar el sistema DMX de **Latencia Afectiva**
> **Hardware:** Raspberry Pi 4 + pantalla DSI 800×480 + Enttec Open DMX (FTDI) + tiras RGB DMX512 + Logitech C922 Pro + Logitech Unifying (MX Keys + MX 3)
> **Autora:** Luma Escalpelo — Octubre 2025

---

## 🧬 Descripción general

El sistema utiliza **Open Lighting Architecture (OLA)** para controlar salidas DMX mediante un **ENTTEC Open DMX USB (FT232R)**.
El flujo general:

```
[ Raspberry Pi 4 ]
   │
   ├── Cámara Logitech C922 Pro
   ├── Pantalla DSI 800×480 (táctil deshabilitada)
   ├── ENTTEC Open DMX USB (FTDI FT232R)
   └── Receptor Logitech Unifying (teclado + mouse)
```

OLA transmite en el **Universe 0**, controlando los canales **22 – 56**.

---

## 🧱 Instalación limpia de OLA

### 1 · Instalar dependencias

```bash
sudo apt update
sudo apt install -y ola libola-dev swig python3-dev build-essential pkg-config
```

### 2 · Habilitar servicio OLA

```bash
sudo systemctl enable --now olad
```

Verificar:

```bash
systemctl status olad --no-pager
```

**Salida esperada**

```
Active: active (running)
... OLA Daemon version 0.10.9
```

---

## 🔌 Conexión del dispositivo ENTTEC

### 3 · Verificar detección FTDI

```bash
lsusb | grep -i ftdi
```

**Salida esperada**

```
Bus 001 Device 004: ID 0403:6001 Future Technology Devices International, Ltd FT232 Serial (UART) IC
```

---

### 4 · Crear alias /dev/opendmx0

```bash
sudo tee /etc/udev/rules.d/60-opendmx.rules > /dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", GROUP="plugdev", MODE="0660", SYMLINK+="opendmx0"
EOF

sudo udevadm control --reload
sudo udevadm trigger
```

Comprobar:

```bash
ls -l /dev/opendmx0
```

**Salida esperada**

```
lrwxrwxrwx 1 root root 7 ... /dev/opendmx0 -> ttyUSB0
```

---

## ⚙️ Configuración OLA

### 5 · Confirmar plugin FTDI

```bash
ola_plugin_info | grep -A2 FTDI
```

Debe mostrar:

```
13  FTDI USB DMX
```

---

### 6 · Habilitar FTDI y desactivar USB conflictivos

```bash
sudo sed -i 's/^enabled *= *.*/enabled = false/' /etc/ola/ola-opendmx.conf
sudo sed -i 's/^enabled *= *.*/enabled = false/' /etc/ola/ola-usbdmx.conf
sudo sed -i 's/^enabled *= *.*/enabled = true/'  /etc/ola/ola-ftdidmx.conf
```

Forzar dispositivo:

```bash
sudo sed -i 's|^device *=.*|device = /dev/opendmx0|' /etc/ola/ola-ftdidmx.conf || echo 'device = /dev/opendmx0' | sudo tee -a /etc/ola/ola-ftdidmx.conf
```

---

### 7 · Reiniciar OLA

```bash
sudo systemctl restart olad
sleep 2
journalctl -u olad --no-pager | tail -n 60
```

**Salida esperada**

```
Trying to start FTDI USB DMX
Found FTDI device: FT232R USB UART
Successfully added 1/1 interfaces.
Installed device: FT232R USB UART serial B001164M
Granularity for FTDI thread is GOOD
```

---

### 8 · Patch al Universe 0

```bash
ola_dev_info | grep -A3 FT232
```

Si no aparece `patched to universe 0`:

```bash
ola_patch --patch --device 8 --port 1 --universe 0
```

Confirmar:

```bash
ola_dev_info | grep -A3 FT232
```

Debe mostrar:

```
Device 8: FT232R USB UART ... patched to universe 0
```

---

## 💡 Prueba funcional DMX

### 9 · Apagar todo

```bash
ola_streaming_client -u 0 -d 1-512=0
```

### 10 · Encender rango activo (22–56)

#### Método continuo (bash)

```bash
while :; do
  ola_streaming_client -u 0 -d 22-56=255 >/dev/null 2>&1
done
```

Ctrl+C para detener.

#### Método Python (estable y preciso)

Archivo: `dmx_full_22_56.py`

```python
from ola.ClientWrapper import ClientWrapper
import time
UNIVERSE = 0
DMX_LOW, DMX_HIGH = 22, 56
FPS = 44

def send():
    data = bytearray(512)
    for ch in range(DMX_LOW, DMX_HIGH + 1):
        data[ch-1] = 255
    wrapper.Client().SendDmx(UNIVERSE, bytes(data), lambda s: None)
    wrapper.AddEvent(int(1000/FPS), send)

wrapper = ClientWrapper()
send()
wrapper.Run()
```

Ejecutar:

```bash
python3 dmx_full_22_56.py
```

---

## 📊 Verificación

En la web de OLA (`http://<ip>:9090`):

1. Abre **DMX Console** o **DMX Monitor**.
2. Verifica que los canales **22–56** están a **255**.
3. El LED del decodificador debe **parpadear rápido**.
4. Las tiras deben encenderse a **brillo completo**.

---

## 🩺 Troubleshooting

| Síntoma                                    | Posible causa                               | Solución                                               |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------ |
| LED del decodificador queda a medio brillo | Transmisión DMX no continua                 | Ejecutar script Python o bash loop                     |
| LED parpadea pero no hay luz               | Invertidos A/B en conector DMX              | Intercambiar pines 2 y 3                               |
| No aparece “FTDI USB DMX” en logs          | Plugin incorrecto o OLA no reiniciado       | Revisar `/etc/ola/ola-ftdidmx.conf` y reiniciar `olad` |
| Cámara deja de funcionar                   | Ninguno de estos pasos afecta `/dev/video*` | Verificar con `v4l2-ctl --list-devices`                |
| Se encienden colores erróneos              | Orden de canales incorrecto (RGB→BRG)       | Ajustar secuencia en código o drivers                  |
| OLA Web muestra menos brillo que terminal  | Fusión HTP con "DMX Console"                | Cierra pestaña DMX Console o asegýate que esté en 255  |

---

## ✅ Resultado final esperado

* Servicio `olad` activo y estable (`systemctl status olad`).
* En `ola_dev_info`: `FT232R USB UART ... patched to universe 0`.
* En `DMX Monitor`: canales **22–56** en **255**.
* LED del driver parpadeando rápido.
* Tiras RGB encendidas a brillo total.
* Cámara y periféricos funcionales.

---

## 🦯 Paths importantes

| Elemento             | Ruta                                 |             |
| -------------------- | ------------------------------------ | ----------- |
| Configuración OLA    | `/etc/ola/`                          |             |
| Regla de dispositivo | `/etc/udev/rules.d/60-opendmx.rules` |             |
| Alias dispositivo    | `/dev/opendmx0`                      |             |
| Logs recientes       | `journalctl -u olad --no-pager       | tail -n 60` |
| Interfaz web         | `http://<ip_de_tu_pi>:9090`          |             |

---

## 🛉 Comandos de mantenimiento

Reiniciar servicio:

```bash
sudo systemctl restart olad
```

Ver logs:

```bash
journalctl -u olad --no-pager | tail -n 50
```

Ver patch actual:

```bash
ola_dev_info
```

---

## 📦 Resumen de configuración final

| Concepto              | Valor                                |
| --------------------- | ------------------------------------ |
| Plugin activo         | `FTDI USB DMX`                       |
| Device                | `/dev/opendmx0`                      |
| Universe              | `0`                                  |
| Rango DMX activo      | `22–56`                              |
| FPS                   | `≈44`                                |
| Brillo máximo         | `255`                                |
| Método de transmisión | Stream continuo (OLA Python Wrapper) |
| Cámara                | Logitech C922 Pro (`/dev/video0`)    |
| Pantalla              | DSI 800×480 (touch deshabilitado)    |

---

### ✍️ Notas de Luma

> 🔮 Si enciendo este archivo dentro de seis meses, quiero recordar que el LED a medio brillo siempre significa que el FTDI no está transmitiendo frames de manera continua.
> Cuando todo funciona, la tira parpadea al compás del universo.

**Nota: Creado con ayuda de ChatGPT5**