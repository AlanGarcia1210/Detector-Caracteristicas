import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 1. Lista de imágenes de Pokémon a evaluar (debemos ajustar la ruta dependiendo de la imagen)
imagenes_pokemon = ["Charizard.png"]

# 2. Inicializar los detectores/descriptores
detectores = {
    "SIFT": cv2.SIFT_create(),
    "ORB": cv2.ORB_create(nfeatures=500),
    "AKAZE": cv2.AKAZE_create()
}

def analizar_pokemon(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"No se encontró la imagen: {ruta_imagen}")
        return

    # Leer imagen en color y convertir a escala de grises
    img_bgr = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    nombre = os.path.basename(ruta_imagen)
    print(f"\n {nombre}")
    print(f"{'Algoritmo':<10} | {'Keypoints':<10} | {'Forma Descriptor':<18} | {'Tiempo (ms)':<12}")
    print("-" * 60)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original\n({nombre})")
    axes[0].axis("off")

    for idx, (nombre_det, detector) in enumerate(detectores.items(), start=1):
        inicio = time.perf_counter()
        
        # Detección de keypoints y extracción de descriptores
        kp, des = detector.detectAndCompute(gris, None)
        
        tiempo_ms = (time.perf_counter() - inicio) * 1000
        forma_des = str(des.shape) if des is not None else "None"

        print(f"{nombre_det:<10} | {len(kp):<10} | {forma_des:<18} | {tiempo_ms:<12.2f}")

        # Dibujar keypoints con tamaño y orientación (RICH_KEYPOINTS)
        img_kp = cv2.drawKeypoints(
            img_rgb, kp, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        axes[idx].imshow(img_kp)
        axes[idx].set_title(f"{nombre_det}\n({len(kp)} keypoints | {tiempo_ms:.1f} ms)")
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(f"resultado_{os.path.splitext(nombre)[0]}.png", dpi=150)
    plt.show()

# Ejecutar el análisis para cada Pokémon
for ruta in imagenes_pokemon:
    analizar_pokemon(ruta)
