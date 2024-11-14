# Person Counter

Un script en Python que captura la imagen de la camara web o un video, y utilizando las bibliotecas YOLO y Sort determina la cantidad de personas que han pasado a través de una linea imaginaria definida por el usuario y les asigna un identificador único.

## Instalacion

### Prerequisitos
- Python 3.10 con soporte para Tk

### Procedimiento
Primero clonamos el repositorio con todos los submodulos necesarios. Para esto usamos:
```
git clone --recurse-submodules https://github.com/DystopianRescuer/PeopleCounter
```
Luego instalamos las dependencias necesarias. La mayoría se pueden instalar con el posterior pip install excepto por lap, para la cual debemos copiarla a mano a nuestro site-packages. Para saber la ubicación de nuestro site-packages usamos:
```
python -m site --user-site
```
Ahi tendremos que copiar todo el contenido de Lap-pkg-for-YOLO/Lap 0.4.0 full package, que son dos carpetas.
Luego, finalmente podemos instalar el resto de dependencias necesarias, usando:
```
pip install -r requirements.txt
```

Una vez hecho esto, podemos correr el programa y ver la hoja de ayuda con:
```
chmod +x pc.py
./pc.py

Una vez abierto el programa con la opción seleccionada, podemos trazar la linea con el mouse, y una vez trazada, presionar q para comenzar el trackeo.

Para salir del programa presionamos q otra vez
