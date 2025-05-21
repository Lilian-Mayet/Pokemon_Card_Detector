# card_warper.py
import cv2
import numpy as np

# Dimensions standard pour la carte redressée (en pixels)
# Le ratio est 6.3cm (largeur) / 8.8cm (hauteur)
POKEMON_CARD_STD_WIDTH = 250  # Largeur cible en pixels
POKEMON_CARD_STD_ASPECT_RATIO = 6.3 / 8.8
POKEMON_CARD_STD_HEIGHT = int(POKEMON_CARD_STD_WIDTH / POKEMON_CARD_STD_ASPECT_RATIO)

def reorder_corners(points):
    """Réorganise un tableau de 4 points dans l'ordre : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    if points.ndim == 3 and points.shape[1] == 1:
        points = points.reshape((4, 2))
    elif points.shape != (4,2):
        # Tenter de convertir si c'est une liste de listes ou de tuples
        try:
            points = np.array(points, dtype="float32").reshape(4,2)
        except:
            raise ValueError("Les points doivent être un tableau 4x2 ou 4x1x2, ou convertible.")


    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect

def warp_card_to_standard_ratio(image, corners_on_original_image):
    """
    Redresse une carte détectée dans une image à ses dimensions standard.

    Args:
        image (numpy.ndarray): L'image originale complète (BGR).
        corners_on_original_image (numpy.ndarray): Tableau 4x2 des coins de la carte
                                                    sur l'image originale.

    Returns:
        numpy.ndarray: L'image de la carte redressée, ou None en cas d'erreur.
    """
    if image is None or corners_on_original_image is None or len(corners_on_original_image) != 4:
        print("Erreur : Entrées invalides pour warp_card_to_standard_ratio.")
        return None

    try:
        ordered_corners = reorder_corners(corners_on_original_image.astype(np.float32))
    except ValueError as e:
        print(f"Erreur lors de la réorganisation des coins : {e}")
        return None


    # Déterminer si la carte est en mode portrait ou paysage en fonction des coins
    (tl, tr, br, bl) = ordered_corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    avg_width = (width_top + width_bottom) / 2

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    avg_height = (height_left + height_right) / 2

    target_w, target_h = POKEMON_CARD_STD_WIDTH, POKEMON_CARD_STD_HEIGHT
    if avg_width > avg_height: # Probablement en mode paysage
        target_w, target_h = POKEMON_CARD_STD_HEIGHT, POKEMON_CARD_STD_WIDTH # Inverser pour paysage

    pts_dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    try:
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, pts_dst)
        warped_image = cv2.warpPerspective(image, transform_matrix, (target_w, target_h))
        return warped_image
    except Exception as e:
        print(f"Erreur lors de la transformation de perspective : {e}")
        return None

if __name__ == "__main__":
    # Test de la fonction de redressement
    # Vous aurez besoin d'une image et des coins d'une carte détectée par card_detector_cv.py
    from detect_card import detect_card_contours # Pour tester
    import os

    test_image_path = "data_for_testing/binder1.png" # Ou une image avec une seule carte bien visible

    image_to_test = cv2.imread(test_image_path)

    if image_to_test is not None:
        print(f"Détection des cartes dans '{test_image_path}' pour le test de redressement...")
        # Utiliser une faible hauteur de redimensionnement pour un traitement plus rapide si l'image est grande
        detected_corners_list, _ = detect_card_contours(image_to_test, resize_height=600)

        if detected_corners_list:
            print(f"{len(detected_corners_list)} carte(s) détectée(s). Redressement de la première...")
            first_card_corners = detected_corners_list[0]

            # Dessiner les coins détectés sur l'original pour vérification
            test_display_img = image_to_test.copy()
            cv2.drawContours(test_display_img, [first_card_corners.astype(np.int32)], -1, (0,255,0), 2)
            cv2.imshow("Carte Détectée (pour redressement)", cv2.resize(test_display_img, (640,480)))


            warped = warp_card_to_standard_ratio(image_to_test, first_card_corners)
            if warped is not None:
                cv2.imshow("Carte Redressée", warped)
                print("Carte redressée avec succès.")
            else:
                print("Échec du redressement de la carte.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Aucune carte détectée pour tester le redressement.")
    else:
        print(f"Impossible de charger l'image de test {test_image_path}")