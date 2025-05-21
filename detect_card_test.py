# enhanced_card_detector.py
import cv2
import numpy as np
import os # Pour le test

# Ratios et tolérances comme dans card_detector_cv.py
POKEMON_CARD_ASPECT_RATIO_PORTRAIT = 6.3 / 8.8
POKEMON_CARD_ASPECT_RATIO_LANDSCAPE = 8.8 / 6.3
ASPECT_RATIO_TOLERANCE = 0.18 # Légèrement augmenté pour plus de flexibilité avec l'excentricité

# Seuils pour les nouveaux filtres (à ajuster expérimentalement)
MIN_ECCENTRICITY = 0.65 # Les rectangles sont assez excentriques. e = sqrt(1 - (b/a)^2) pour une ellipse
MAX_ECCENTRICITY = 0.99 # Pas une ligne pure
MIN_CONVEXITY_RATIO = 0.92 # Doit être proche de 1 pour les formes rectangulaires/convexes

def calculate_eccentricity(contour):
    """Calcule l'excentricité d'un contour en l'ajustant à une ellipse."""
    if len(contour) < 5: # cv2.fitEllipse a besoin d'au moins 5 points
        return 0 # Ne peut pas calculer, retourne une valeur qui sera probablement filtrée

    try:
        ellipse = cv2.fitEllipse(contour)
        # ellipse est ((center_x, center_y), (minor_axis, major_axis), angle)
        minor_axis, major_axis = ellipse[1] # minor_axis et major_axis peuvent être intervertis

        # S'assurer que major_axis est bien le plus grand
        if minor_axis > major_axis:
            minor_axis, major_axis = major_axis, minor_axis

        if major_axis == 0: # Éviter la division par zéro
            return 1 # Tendance vers une ligne si l'axe majeur est nul (improbable pour de vrais contours)

        # Excentricité e = sqrt(1 - (b/a)^2)
        eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2)
        return eccentricity
    except cv2.error as e:
        # Parfois, fitEllipse échoue sur des contours dégénérés
        # print(f"cv2.error in fitEllipse: {e} for contour with {len(contour)} points")
        return 0 # Ou une autre valeur indiquant un échec


def calculate_convexity_ratio(contour):
    """Calcule le rapport entre le périmètre du contour et le périmètre de son enveloppe convexe."""
    if len(contour) < 3: # Besoin d'au moins 3 points pour une enveloppe convexe
        return 0

    try:
        hull = cv2.convexHull(contour)
        contour_perimeter = cv2.arcLength(contour, True)
        hull_perimeter = cv2.arcLength(hull, True)

        if hull_perimeter == 0: # Éviter la division par zéro
            return 0

        convexity_ratio = contour_perimeter / hull_perimeter
        return convexity_ratio
    except Exception as e:
        # print(f"Error calculating convexity: {e}")
        return 0


def detect_enhanced_card_contours(image_bgr, resize_height=700):
    """
    Détecte les contours de cartes potentielles dans une image en utilisant des techniques améliorées.

    Args:
        image_bgr (numpy.ndarray): L'image d'entrée (couleur BGR).
        resize_height (int): Hauteur pour redimensionner l'image pour traitement. None pour ne pas redimensionner.

    Returns:
        list: Une liste de tableaux de coins (chaque tableau a 4 points [x, y])
              pour chaque carte détectée, mis à l'échelle des dimensions de l'image originale.
        numpy.ndarray: L'image traitée (redimensionnée si resize_height est défini) avec les contours dessinés (pour débogage).
    """
    if image_bgr is None:
        print("Erreur : Image d'entrée non valide.")
        return [], None

    orig_h, orig_w = image_bgr.shape[:2]
    processed_image = image_bgr.copy() # Image qui sera modifiée pour le traitement
    scale = 1.0

    if resize_height:
        scale = resize_height / orig_h
        processing_width = int(orig_w * scale)
        processed_image = cv2.resize(image_bgr, (processing_width, resize_height))

    # 1. Pré-traitement
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Un léger flou initial

    # 2. Détection des bords (Canny est un bon point de départ)
    edged = cv2.Canny(blurred, 30, 100) # Seuil bas pour attraper les bords faibles

    # 3. Opération Morphologique de Fermeture (Closing)
    # L'article utilise un disque de rayon 7. Un élément elliptique de taille 11x11 ou 15x15 pourrait s'en approcher.
    # Une fermeture plus importante peut aider à combler les détails internes des cartes
    # et à rendre la forme globale plus solide.
    kernel_closing_size = 11 # Doit être impair. Ex: 7, 9, 11, 15. À tester.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_closing_size, kernel_closing_size))
    closed_edges = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2) # iterations=2 pour une fermeture plus forte

    # 4. Binarisation (Canny et closing produisent déjà une image binaire, mais on peut ré-appliquer si besoin)
    # Pour cet exemple, 'closed_edges' est déjà binaire.

    # 5. Trouver les contours sur l'image après fermeture
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrage des contours
    min_area_ratio = 0.008 # % de l'aire de l'image traitée
    max_area_ratio = 0.6   # % de l'aire de l'image traitée
    min_contour_area = min_area_ratio * processed_image.shape[0] * processed_image.shape[1]
    max_contour_area = max_area_ratio * processed_image.shape[0] * processed_image.shape[1]

    detected_card_scaled_corners = []
    debug_contours_to_draw = []

    # print(f"Total contours found: {len(contours)}")

    for i, cnt in enumerate(contours):
        # Filtre par Aire
        area = cv2.contourArea(cnt)
        if not (min_contour_area < area < max_contour_area):
            # print(f"Contour {i} rejeté par aire: {area:.0f}")
            continue

        # Approximation en polygone pour trouver les 4 coins
        peri = cv2.arcLength(cnt, True)
        approx_corners = cv2.approxPolyDP(cnt, 0.03 * peri, True) # Epsilon typique: 0.02-0.04

        if len(approx_corners) != 4:
            # print(f"Contour {i} rejeté par nombre de coins: {len(approx_corners)}")
            continue

        # Filtre par Ratio d'Aspect de la boîte englobante
        (x_br, y_br, w_br, h_br) = cv2.boundingRect(approx_corners)
        if h_br == 0: continue
        aspect_ratio_br = float(w_br) / h_br
        is_portrait = (POKEMON_CARD_ASPECT_RATIO_PORTRAIT * (1 - ASPECT_RATIO_TOLERANCE) < aspect_ratio_br < POKEMON_CARD_ASPECT_RATIO_PORTRAIT * (1 + ASPECT_RATIO_TOLERANCE))
        is_landscape = (POKEMON_CARD_ASPECT_RATIO_LANDSCAPE * (1 - ASPECT_RATIO_TOLERANCE) < aspect_ratio_br < POKEMON_CARD_ASPECT_RATIO_LANDSCAPE * (1 + ASPECT_RATIO_TOLERANCE))

        if not (is_portrait or is_landscape):
            # print(f"Contour {i} rejeté par ratio de boîte englobante: {aspect_ratio_br:.2f}")
            continue

        # Filtre par Excentricité
        eccentricity = calculate_eccentricity(cnt)
        if not (MIN_ECCENTRICITY < eccentricity < MAX_ECCENTRICITY):
            # print(f"Contour {i} rejeté par excentricité: {eccentricity:.2f} (Attendu: {MIN_ECCENTRICITY}-{MAX_ECCENTRICITY})")
            continue

        # Filtre par Ratio de Convexité
        convexity_ratio = calculate_convexity_ratio(cnt)
        if convexity_ratio < MIN_CONVEXITY_RATIO : # Doit être >= MIN_CONVEXITY_RATIO et <= 1.0
            # print(f"Contour {i} rejeté par convexité: {convexity_ratio:.2f} (Attendu: >{MIN_CONVEXITY_RATIO})")
            continue

        # Si tous les filtres sont passés :
        # print(f"Contour {i} ACCEPTÉ : Aire={area:.0f}, Coins={len(approx_corners)}, AR={aspect_ratio_br:.2f}, Ecc={eccentricity:.2f}, Conv={convexity_ratio:.2f}")
        scaled_back_corners = (approx_corners / scale).astype(np.int32)
        detected_card_scaled_corners.append(scaled_back_corners)
        debug_contours_to_draw.append(approx_corners)


    # Dessiner les contours finaux sur une image de débogage
    output_debug_image = processed_image.copy()
    cv2.drawContours(output_debug_image, debug_contours_to_draw, -1, (0, 255, 0), 2)
    # Afficher aussi l'image des bords fermés pour le débogage
    # cv2.imshow("Closed Edges (for debug)", closed_edges)

    return detected_card_scaled_corners, output_debug_image


if __name__ == "__main__":
    # Test de la fonction de détection améliorée
    test_image_path = "data_for_testing/binder1.png" # Image avec plusieurs cartes
    # test_image_path = "data_for_testing/charizard.jpg" # Image avec une seule carte
    # test_image_path = "dummy_perspective_card.png" # Image de test simple

    if not os.path.exists(test_image_path) and test_image_path == "dummy_perspective_card.png":
        dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
        pts = np.array([[150,100], [450,120], [400,500], [100,480]], np.int32)
        cv2.fillPoly(dummy_image, [pts], (200,200,200))
        cv2.imwrite("dummy_perspective_card.png", dummy_image)
    elif not os.path.exists(test_image_path):
        print(f"Chemin de test non trouvé : {test_image_path}. Veuillez fournir une image valide.")
        exit()

    image_to_test_enh = cv2.imread(test_image_path)
    if image_to_test_enh is None:
        print(f"Erreur: Impossible de charger l'image de test depuis {test_image_path}")
    else:
        print(f"Test de la détection améliorée sur {test_image_path}...")
        # Utiliser une hauteur de redimensionnement plus faible pour des tests plus rapides
        card_corners_list_enh, debug_img_enh = detect_enhanced_card_contours(image_to_test_enh, resize_height=800)
        print(f"{len(card_corners_list_enh)} carte(s) potentielle(s) détectée(s) avec la méthode améliorée.")

        if debug_img_enh is not None:
            # Redimensionner l'original pour qu'il corresponde à la hauteur de l'image de débogage pour l'affichage
            h_debug, w_debug = debug_img_enh.shape[:2]
            display_orig = cv2.resize(image_to_test_enh, (int(image_to_test_enh.shape[1] * (h_debug/image_to_test_enh.shape[0])), h_debug) )

            cv2.imshow("Original (pour comparaison)", display_orig)
            cv2.imshow("Détection Améliorée (Contours verts sur image traitée)", debug_img_enh)

            orig_with_scaled_contours_enh = image_to_test_enh.copy()
            for corners_enh in card_corners_list_enh:
                 cv2.drawContours(orig_with_scaled_contours_enh, [corners_enh], -1, (255,0,255), 3) # Magenta
            cv2.imshow("Contours détectés (Amélioré) sur l'image originale", orig_with_scaled_contours_enh)

            cv2.waitKey(0)
            cv2.destroyAllWindows()