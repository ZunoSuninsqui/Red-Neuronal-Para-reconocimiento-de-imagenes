# Reconocimiento de dígitos dibujados en canvas

Este repositorio contiene una red neuronal convolucional para reconocer dígitos (0-9) dibujados a mano en un canvas de 200×200 píxeles.  El proyecto incluye:

* Pipeline de preprocesamiento unificado para entrenamiento e inferencia.
* Modelo CNN con normalización por lotes, Dropout y regularización L2.
* Entrenamiento reproducible con partición entrenamiento/validación estratificada y *early stopping*.
* Calibración de probabilidades mediante *temperature scaling* y métricas de confiabilidad (ECE, diagramas de fiabilidad).
* Scripts CLI para entrenar (`train.py`), evaluar (`eval.py`) e inferir (`infer.py`).
* Pruebas automatizadas para validar las transformaciones clave (`tests/test_pipeline.py`).

## Requisitos

Crea un entorno virtual y instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Nota:** TensorFlow requiere librerías adicionales según tu sistema operativo. Consulta la [documentación oficial](https://www.tensorflow.org/install) si encuentras problemas de instalación.

## Estructura principal

```
configs/                # Configuraciones YAML para entrenamiento/inferencia
modelos/                # Checkpoints, métricas y artefactos generados
tests/                  # Pruebas automatizadas del pipeline
utils/                  # Transformaciones y calibración compartidas
Dataset/                # Dataset con subcarpetas por estudiante (no se versiona)
train.py                # Script de entrenamiento
infer.py                # CLI de inferencia para imágenes nuevas
eval.py                 # Evaluación y métricas adicionales
canvas_app.py           # Interfaz gráfica opcional basada en Tk
classes.json            # Mapeo estable de etiquetas
```

## Entrenamiento

1. Verifica que la carpeta `Dataset/` mantenga la estructura original (`<estudiante>/imagenes/*.png`).
2. Ajusta los hiperparámetros en `configs/default.yaml` si lo necesitas.
3. Ejecuta el entrenamiento:

   ```bash
   python train.py --config configs/default.yaml
   ```

   El script:

   * Aplica los *transforms* definidos en `utils/transforms.py` (inversión automática, centrado, padding, normalización).
   * Divide los datos en entrenamiento/validación (80/20) con semilla fija (`seed` en la configuración).
   * Entrena la CNN con *early stopping* y `ReduceLROnPlateau`.
   * Guarda el mejor modelo, métricas (`modelos/metricas_entrenamiento.json`), estadísticas del dataset (`modelos/dataset_stats.json`), matriz de confusión y diagrama de fiabilidad.
   * Ajusta la temperatura óptima (`modelos/calibration_temperature.txt`) y calcula la ECE antes/después de calibrar.

## Evaluación

Reproduce las métricas en el conjunto de validación (se usa la misma semilla para recrear el `split`):

```bash
python eval.py --config configs/default.yaml
```

El resultado se almacena en `modelos/eval_metrics.json` e incluye exactitud, F1 macro y ECE calibrado (si existe la temperatura guardada).

## Inferencia

Clasifica una imagen externa (PNG/JPG). Se aceptan imágenes con fondo oscuro o claro; la tubería ajusta la inversión automáticamente.

```bash
python infer.py path/a/la_imagen.png --topk 5
```

Opciones relevantes:

* `--no-calibration`: deshabilita la corrección por temperatura.
* `--raw`: devuelve un JSON con las probabilidades `top-k` (ideal para integraciones).
* `--dataset-stats`, `--calibration`, `--model`: permiten usar rutas personalizadas para los artefactos producidos por `train.py`.

El script detecta dibujos casi vacíos (energía baja) y avisa cuando la predicción podría no ser fiable.

## Interfaz gráfica

Tras entrenar el modelo puedes abrir la interfaz de canvas original:

```bash
python main.py --solo-interfaz
```

La interfaz reutiliza el preprocesamiento actualizado y las probabilidades calibradas para mostrar predicciones más confiables.

## Pruebas

Ejecuta la batería de pruebas rápidas para validar el pipeline de datos y la calibración:

```bash
pytest
```

> Algunas pruebas requieren `Pillow`; si no está instalado, se marcarán como omitidas.

## Métricas esperadas

Los resultados concretos dependen de la distribución exacta del dataset, pero como referencia en el conjunto de validación se esperan:

* Exactitud ≥ 0.97
* F1 macro ≥ 0.97
* Expected Calibration Error (ECE) < 0.05 tras la calibración

Los artefactos `modelos/confusion_matrix.png` y `modelos/reliability_diagram.png` permiten inspeccionar visualmente la calidad del modelo.

## Troubleshooting

* **Predicciones con confianza muy alta pero incorrectas:** asegúrate de usar `infer.py` (o la interfaz) después de ejecutar `train.py`, ya que el preprocesamiento y la calibración se actualizan allí.
* **Imagen invertida:** la función de preprocesado invierte automáticamente cuando detecta fondos claros. Si observas resultados extraños, revisa la imagen de entrada guardando la versión normalizada (puedes modificar `infer.py` para depurar).
* **Errores de TensorFlow/Pillow:** instala las dependencias listadas en `requirements.txt` y verifica tu entorno virtual.
