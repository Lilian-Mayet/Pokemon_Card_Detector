import json
from PIL import Image
import imagehash # Bibliothèque pour le hachage perceptuel
import os

# Doit correspondre au fichier JSON généré
HASHED_CARDS_JSON_PATH = "pokemon_card_hashes.json" # Ou votre fichier JSON pertinent

def load_hashed_data(json_path):
    """Charge les données de hachage depuis le fichier JSON."""
    try:
        with open(json_path, 'r') as f:
            hashed_data = json.load(f)
        for card_entry in hashed_data:
            card_entry['hash_obj'] = imagehash.hex_to_hash(card_entry['hash'])
        return hashed_data
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON de hachage '{json_path}' n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le JSON '{json_path}'.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données de hachage : {e}")
        return None

def custom_hex_char_diff_distance(hex_hash1_str, hex_hash2_str):
    """
    Calcule une distance personnalisée basée sur la somme des différences
    absolues des valeurs des caractères hexadécimaux.
    """
    if len(hex_hash1_str) != len(hex_hash2_str):
        return float('inf')
    total_difference = 0
    for char1, char2 in zip(hex_hash1_str, hex_hash2_str):
        try:
            val1 = int(char1, 16)
            val2 = int(char2, 16)
            total_difference += abs(val1 - val2)
        except ValueError:
            return float('inf') # Caractère non hexadécimal
    return total_difference

def find_matching_card_with_tiebreak(input_image_path, hashed_data,
                                     max_hamming_distance=10,
                                     max_custom_distance_for_tiebreak=40): # Seuil pour la distance personnalisée en cas d'égalité
    """
    Trouve la carte la plus correspondante. Utilise la distance de Hamming en priorité.
    En cas d'égalité sur la distance de Hamming, utilise la distance personnalisée
    (différence des caractères hexadécimaux) pour départager.

    Args:
        input_image_path (str): Chemin vers l'image de la carte à identifier.
        hashed_data (list): Liste des dictionnaires de cartes hachées.
        max_hamming_distance (int): Seuil pour la distance de Hamming.
        max_custom_distance_for_tiebreak (int): Seuil pour la distance personnalisée LORSQU'ELLE EST UTILISÉE POUR DÉPARTAGER.

    Returns:
        dict: Le dictionnaire de la carte correspondante ou None.
    """
    if not os.path.exists(input_image_path):
        print(f"Erreur : Le fichier image d'entrée '{input_image_path}' n'a pas été trouvé.")
        return None
    if not hashed_data:
        print("Erreur : Aucune donnée de hachage fournie.")
        return None

    try:
        input_image = Image.open(input_image_path)
        input_hash_obj = imagehash.phash(input_image)
        input_hash_hex_str = str(input_hash_obj)
        print(f"\nHachage de l'image d'entrée '{os.path.basename(input_image_path)}': {input_hash_hex_str}")
    except Exception as e:
        print(f"Erreur lors du traitement de l'image d'entrée '{input_image_path}': {e}")
        return None

    # Étape 1: Calculer toutes les distances de Hamming et trouver le minimum
    min_hamming_dist = float('inf')
    potential_hamming_matches = [] # Va stocker les (distance_hamming, card_entry)

    for card_entry in hashed_data:
        stored_hash_obj = card_entry.get('hash_obj')
        if not stored_hash_obj:
            continue

        hamming_dist = input_hash_obj - stored_hash_obj
        potential_hamming_matches.append((hamming_dist, card_entry))

        if hamming_dist < min_hamming_dist:
            min_hamming_dist = hamming_dist

    # Filtrer pour ne garder que les cartes ayant la distance de Hamming minimale
    tied_on_hamming = [entry for dist, entry in potential_hamming_matches if dist == min_hamming_dist]

    # Si la meilleure distance de Hamming est déjà trop élevée, pas besoin d'aller plus loin
    if min_hamming_dist > max_hamming_distance:
        print(f"\nAucune correspondance satisfaisante. Distance de Hamming minimale ({min_hamming_dist}) > seuil ({max_hamming_distance}).")
        if tied_on_hamming: # Il y avait des "meilleures" mais trop loin
             print(f"(La/les plus proche(s) par Hamming était/étaient : {[c['name'] for c in tied_on_hamming]} avec distance {min_hamming_dist})")
        return None

    # Étape 2: Gérer les résultats
    final_best_match = None
    final_distance_metric = ""
    final_distance_value = float('inf')

    if len(tied_on_hamming) == 1:
        final_best_match = tied_on_hamming[0]
        final_distance_metric = "Hamming"
        final_distance_value = min_hamming_dist
        print(f"\nCorrespondance unique trouvée par distance de Hamming.")
    elif len(tied_on_hamming) > 1:
        print(f"\nÉgalité sur la distance de Hamming ({min_hamming_dist}) pour {len(tied_on_hamming)} cartes : {[c['name'] for c in tied_on_hamming]}.")
        print("Utilisation de la distance personnalisée (différence des caractères hex) pour départager...")

        min_custom_dist_for_tiebreak = float('inf')
        tiebreak_winner = None

        for card_entry_tied in tied_on_hamming:
            stored_hash_hex_str = card_entry_tied.get('hash')
            if not stored_hash_hex_str:
                continue

            custom_dist = custom_hex_char_diff_distance(input_hash_hex_str, stored_hash_hex_str)
            # print(f"  - Carte départage: '{card_entry_tied['name']}', Dist. perso.: {custom_dist}")


            if custom_dist < min_custom_dist_for_tiebreak:
                min_custom_dist_for_tiebreak = custom_dist
                tiebreak_winner = card_entry_tied
            # En cas d'égalité sur la distance personnalisée, on garde le premier trouvé

        if tiebreak_winner:
            if min_custom_dist_for_tiebreak <= max_custom_distance_for_tiebreak:
                final_best_match = tiebreak_winner
                final_distance_metric = "Personnalisée (après égalité Hamming)"
                final_distance_value = min_custom_dist_for_tiebreak # La distance pertinente ici est la distance personnalisée
                print(f"Départage réussi avec la distance personnalisée.")
            else:
                print(f"Le gagnant du départage ('{tiebreak_winner['name']}') a une distance personnalisée ({min_custom_dist_for_tiebreak}) trop élevée (seuil: {max_custom_distance_for_tiebreak}).")
                # On pourrait décider de ne pas retourner de correspondance ou de retourner la correspondance Hamming si elle était acceptable
                # Pour l'instant, on considère que si le tie-break échoue au seuil, c'est un échec global.
                final_best_match = None # Ou alors on pourrait revenir au fait que la distance Hamming était bonne
        else:
            print("Impossible de départager avec la distance personnalisée (peut-être un problème de données).")
            # On pourrait prendre le premier de la liste `tied_on_hamming` si la dist Hamming est OK.
            # Pour l'instant, on considère cela comme un échec si le tie-break était nécessaire mais n'a pas abouti.
            final_best_match = None

    else: # Aucune carte trouvée (liste `hashed_data` vide ou problème)
        print("\nAucune carte n'a pu être évaluée.")
        return None


    # Afficher le résultat final
    if final_best_match:
        print(f"\n==> Meilleure correspondance finale : '{final_best_match['name']}' (ID: {final_best_match['id']})")
        print(f"    Métrique déterminante : {final_distance_metric}")
        if final_distance_metric == "Personnalisée (après égalité Hamming)":
            print(f"    Distance de Hamming (commune aux égalités) : {min_hamming_dist}")
            print(f"    Distance personnalisée (utilisée pour départager) : {final_distance_value}")
        else: # Hamming était unique
             print(f"    Distance de Hamming : {final_distance_value}")
        return final_best_match
    else:
        print("\n==> Aucune correspondance finale satisfaisante trouvée après toutes les étapes.")
        return None


if __name__ == "__main__":
    all_hashed_cards = load_hashed_data(HASHED_CARDS_JSON_PATH)

    if all_hashed_cards:
        # --- Configuration du test ---
        # Mettez ici le chemin vers l'image que vous voulez tester
        input_image_file = "data_for_testing/pharamp.png" # Assurez-vous que ce chemin est correct

        # Pour simuler une égalité, vous pourriez avoir besoin de modifier manuellement
        # le fichier JSON pour que deux cartes aient des hachages pHash très proches
        # d'une image de test, ou que l'image de test soit pile entre deux hachages.
        # C'est difficile à garantir sans données spécifiques.

        if not os.path.exists(input_image_file):
            print(f"Fichier de test '{input_image_file}' non trouvé. Veuillez vérifier le chemin.")
        else:
            HAMMING_THRESHOLD = 16
            CUSTOM_TIEBREAK_THRESHOLD = 45 # Seuil pour la distance personnalisée SI elle est utilisée

            print(f"--- Test avec Distance de Hamming (seuil {HAMMING_THRESHOLD}), puis départage personnalisé (seuil {CUSTOM_TIEBREAK_THRESHOLD}) ---")
            matched_card = find_matching_card_with_tiebreak(
                input_image_file,
                all_hashed_cards,
                max_hamming_distance=HAMMING_THRESHOLD,
                max_custom_distance_for_tiebreak=CUSTOM_TIEBREAK_THRESHOLD
            )

            if matched_card:
                print(f"\nCarte identifiée par la fonction : {matched_card['name']}")
            else:
                print("\nAucune carte n'a été identifiée par la fonction.")

            # Vous pouvez ajouter d'autres images de test ici
            # input_image_file_2 = "data_for_testing/autre_carte.jpg"
            # if os.path.exists(input_image_file_2):
            #     print(f"\n--- Test avec {input_image_file_2} ---")
            #     matched_card_2 = find_matching_card_with_tiebreak(
            #         input_image_file_2,
            #         all_hashed_cards,
            #         max_hamming_distance=HAMMING_THRESHOLD,
            #         max_custom_distance_for_tiebreak=CUSTOM_TIEBREAK_THRESHOLD
            #     )
            #     if matched_card_2:
            #         print(f"\nCarte identifiée : {matched_card_2['name']}")
            #     else:
            #         print("\nAucune carte n'a été identifiée.")
    else:
        print("Les données de hachage n'ont pas pu être chargées. Impossible d'exécuter le test.")