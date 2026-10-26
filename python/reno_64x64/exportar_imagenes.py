import numpy as np
import matplotlib.pyplot as plt
import os

# 1. NUEVA ruta del archivo de solución del tucán
archivo = "../results/tvlearn_geo_deblur_N256_k12/sols/sol_geo_deblur_N256_k12_s000.npz"
data = np.load(archivo)

print("Variables encontradas en el archivo:", data.files)

# 2. NUEVO nombre base para los archivos generados
nombre_base = "geo_deblur"
claves_deseadas = ['f', 'u', 'u_clean']

# 3. Buscar y guardar únicamente las imágenes especificadas
for clave in claves_deseadas:
    if clave in data.files:
        matriz = data[clave]
        
        # Si la matriz tiene 2 dimensiones, es una imagen
        if len(matriz.shape) == 2:
            alto, ancho = matriz.shape
            
            # Crear el nombre de la carpeta (ej. tucan_64x64)
            nombre_carpeta = f"{nombre_base}_{alto}x{ancho}"
            os.makedirs(nombre_carpeta, exist_ok=True)
            
            # Guardar la imagen
            nombre_archivo = f"{nombre_base}_{clave}.png"
            ruta_guardado = os.path.join(nombre_carpeta, nombre_archivo)
            
            plt.imsave(ruta_guardado, matriz, cmap="gray")
            print(f"✔ Imagen guardada en: {ruta_guardado}")
    else:
        print(f"⚠ La variable '{clave}' no se encuentra en el archivo .npz y será omitida.")