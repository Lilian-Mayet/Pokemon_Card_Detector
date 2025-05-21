# main_live_scanner.py
import cv2
import numpy as np
from PIL import Image # Pour convertir l'image OpenCV en PIL pour l'identificateur

from detect_card import detect_card_boxes
from card_warper import warp_card_to_standard_ratio
from card_identifier import load_hashed_data, find_matching_card

# --- Paramètres ---
CAMERA_INDEX = 0 # Essayez 0, 1, etc. si votre webcam par défaut n'est pas la bonne
RESIZE_HEIGHT_FOR_DETECTION = 1000 # Hauteur pour le traitement de détection
HAMMING_THRESHOLD = 14
HASHED_CARDS_JSON_PATH = "pokemon_card_hashes.json"
# --- Fin des Paramètres ---

def main():
    # Charger la base de données de hachages au démarrage
    card_hash_database = load_hashed_data(HASHED_CARDS_JSON_PATH)
    if not card_hash_database:
        print("Impossible de charger la base de données de hachages. Arrêt du scanner.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la caméra (index {CAMERA_INDEX}).")
        return

    print("\nScanner en direct démarré. Appuyez sur 'q' pour quitter.")
    print(f"Utilisation des seuils : Hamming <= {HAMMING_THRESHOLD}, Départage Perso <= {CUSTOM_TIEBREAK_THRESHOLD}")

    frame_count = 0
    processing_interval = 3 # Traiter une image sur N pour alléger (ex: 3 ou 5)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire l'image de la caméra.")
            break

        display_frame = frame.copy() # Copie pour l'affichage avec annotations
        frame_count += 1

        if frame_count % processing_interval == 0: # Traiter périodiquement
            # 1. Détecter les contours des cartes
            # La fonction retourne les coins à l'échelle de `frame` (l'image originale de la caméra)
            detected_card_corners_list = detect_card_boxes(frame, resize_height=RESIZE_HEIGHT_FOR_DETECTION)
            print(detected_card_corners_list)

            for corners in detected_card_corners_list:
                # 2. Redresser la carte
                warped_card_cv = warp_card_to_standard_ratio(frame, corners)

                if warped_card_cv is not None:
                    # Convertir l'image redressée (OpenCV BGR) en PIL Image (RGB)
                    try:
                        pil_warped_card = Image.fromarray(cv2.cvtColor(warped_card_cv, cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        print(f"Erreur de conversion OpenCV vers PIL: {e}")
                        continue

                    # 3. Identifier la carte
                    identified_card = find_matching_card(
                        pil_warped_card,
                        card_hash_database,
                        max_hamming_distance=HAMMING_THRESHOLD,
                        #max_custom_distance_for_tiebreak=CUSTOM_TIEBREAK_THRESHOLD
                    )

                    # 4. Afficher les résultats sur `display_frame`
                    # Dessiner le contour de la carte détectée
                    cv2.drawContours(display_frame, [corners.astype(np.int32)], -1, (0, 255, 0), 2)

                    text_to_display = "Inconnue"
                    text_color = (0, 0, 255) # Rouge pour inconnue

                    if identified_card:
                        text_to_display = f"{identified_card['name']} (ID: {identified_card['id']})"
                        #text_to_display += f"\n{reason}" # Peut être trop long pour l'affichage
                        text_color = (255, 100, 0) # Bleu pour identifiée
                        print(f"  Identifié: {identified_card['name']}")


                    # Afficher le texte près du coin supérieur gauche de la carte détectée
                    # (x,y) du premier coin (après réorganisation, ce sera le coin en haut à gauche)
                    # Pour plus de simplicité, utilisons la boîte englobante des coins pour positionner le texte
                    rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(corners.astype(np.int32))
                    
                    # Mettre le texte sur plusieurs lignes si nécessaire
                    y0, dy = rect_y - 10, 18 # Position initiale Y et espacement vertical
                    for i, line in enumerate(text_to_display.split('\n')):
                        y_pos = y0 + i * dy
                        # S'assurer que le texte ne sort pas en haut de l'écran
                        if y_pos < 15 : y_pos = 15 + i * dy

                        cv2.putText(display_frame, line, (rect_x, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                # else:
                    # print("Échec du redressement d'une carte détectée.")
        else:
            # Si on ne traite pas, on peut quand même redessiner les dernières détections
            # pour une sensation plus fluide, mais cela complexifie.
            # Pour l'instant, on affiche juste le frame sans nouvelles détections.
            pass


        cv2.imshow("Scanner de Cartes Pokémon en Direct", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Arrêt du scanner...")
            break
        elif key == ord(' '): # Barre d'espace pour forcer le traitement
            print("Traitement forcé de l'image...")
            frame_count = 0 # Pour que le prochain `if` soit vrai


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Vérifier si le fichier de hachage existe avant de démarrer
    import os
    if not os.path.exists(HASHED_CARDS_JSON_PATH):
         print(f"ERREUR CRITIQUE : Le fichier de base de données de hachages '{HASHED_CARDS_JSON_PATH}' est introuvable.")
         print("Veuillez exécuter le script de hachage de la base de données d'abord.")
    else:
        main()