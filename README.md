# Reconocimiento de dígitos 1-9

Aplicación de escritorio construida con Tkinter y PyTorch para entrenar y utilizar un modelo de reconocimiento de dígitos escritos a mano (1–9). Permite dibujar en un lienzo, preprocesar la imagen, entrenar una red neuronal con un dataset local y realizar predicciones.

## Requisitos

- Python 3.9 o superior
- Sistema operativo con soporte para Tkinter (Windows, Linux o macOS)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

### Flujo recomendado

1. **Seleccionar dataset**: Usa el botón correspondiente y elige la carpeta raíz de tu dataset local.
2. **Entrenar**: Tras validar la estructura del dataset (clases 1–9 con al menos 20 imágenes por clase), inicia el entrenamiento. El proceso se ejecuta en un hilo separado y los logs aparecen en la sección inferior.
3. **Guardar/Cargar modelo**: Puedes guardar el modelo entrenado (`.pt/.pth`) y cargarlo posteriormente sin necesidad de reentrenar.
4. **Dibujar y predecir**: Traza un dígito en el lienzo, pulsa **Preprocesar** para ver la versión 28×28 y luego **Predecir** para obtener la clase y confianza.

## Estructura del proyecto

- `main.py`: Punto de entrada que inicializa la interfaz.
- `ui.py`: Implementación de la interfaz Tkinter y la lógica de interacción.
- `data_loader.py`: Escaneo del dataset, reglas de etiquetado y creación de `DataLoader`.
- `preprocess.py`: Pipeline de preprocesado compartido entre entrenamiento e inferencia.
- `model.py`: Definición de la CNN (tipo LeNet) y utilidades de serialización.
- `train.py`: Bucle de entrenamiento, métricas y matriz de confusión.
- `predict.py`: Funciones de inferencia para tensores o imágenes PIL.
- `utils.py`: Utilidades generales (logging, dispositivo, semillas).

## Dataset esperado

La aplicación soporta dos estructuras de carpetas:

```
root/1/*.png|jpg|jpeg|bmp
...
root/9/*.png|jpg|jpeg|bmp
```

o bien

```
root/<persona>/imagenes/*.png|jpg|jpeg|bmp
```

Reglas de etiquetado:

- Si el directorio padre inmediato es un dígito 1–9, se usa como etiqueta.
- Si el directorio padre inmediato es `imagenes`, se mira el directorio superior: si es un dígito 1–9 se usa; en caso contrario, se extrae el primer dígito 1–9 del nombre del archivo.
- Archivos que no cumplan las reglas anteriores se ignoran.

Los formatos aceptados son `.png`, `.jpg`, `.jpeg` y `.bmp`. El dataset se divide automáticamente en entrenamiento/validación (80/20) con barajado estratificado.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
