import json
from PIL import Image
import imagehash # Bibliothèque pour le hachage perceptuel
import os

HASHED_CARDS_JSON_PATH = "pokemon_card_hashes.json" # Doit correspondre au fichier de la Partie 1

def load_hashed_data(json_path):
    """Charge les données de hachage depuis le fichier JSON."""
    try:
        with open(json_path, 'r') as f:
            hashed_data = json.load(f)
        # Convertir les chaînes de hachage en objets ImageHash
        for card_entry in hashed_data:
            card_entry['hash_obj'] = imagehash.hex_to_hash(card_entry['hash'])
        return hashed_data
    except FileNotFoundError:
        print(f"Erreur : Le fichier JSON de hachage '{json_path}' n'a pas été trouvé.")
        print("Veuillez d'abord exécuter le script de hachage de la base de données.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur : Impossible de décoder le fichier JSON '{json_path}'. Est-il valide ?")
        return None
    except Exception as e:
        print(f"Une erreur s'est produite lors du chargement des données de hachage : {e}")
        return None


def find_matching_card(input_image_path, hashed_data, max_hamming_distance=10):
    """
    Trouve la carte la plus correspondante dans les données hachées pour une image d'entrée.

    Args:
        input_image_path (str): Chemin vers l'image de la carte à identifier.
        hashed_data (list): Liste des dictionnaires de cartes hachées.
        max_hamming_distance (int): La distance de Hamming maximale pour considérer une correspondance.
                                     Ajustez cette valeur en fonction de la robustesse souhaitée.
                                     Une valeur plus faible est plus stricte.
                                     Pour pHash, des valeurs entre 0 et 10 sont généralement raisonnables.
    Returns:
        dict: Le dictionnaire de la carte correspondante ou None si aucune correspondance satisfaisante n'est trouvée.
    """
    if not os.path.exists(input_image_path):
        print(f"Erreur : Le fichier image d'entrée '{input_image_path}' n'a pas été trouvé.")
        return None

    try:
        input_image = Image.open(input_image_path)
        input_hash = imagehash.phash(input_image) # Calcule le pHash de l'image d'entrée
        print(f"Hachage de l'image d'entrée '{input_image_path}': {input_hash}")
    except FileNotFoundError:
        print(f"Erreur : Le fichier image d'entrée '{input_image_path}' n'a pas été trouvé.")
        return None
    except Exception as e:
        print(f"Erreur lors du traitement de l'image d'entrée '{input_image_path}': {e}")
        return None

    best_match = None
    smallest_distance = float('inf')

    for card_entry in hashed_data:
        stored_hash_obj = card_entry.get('hash_obj')
        if not stored_hash_obj: # S'assure que hash_obj a été correctement chargé
            print(f"Attention : Entrée de carte sans objet de hachage valide : {card_entry.get('name')}")
            continue

        # Calcule la distance de Hamming
        # La bibliothèque imagehash surcharge l'opérateur de soustraction pour cela
        distance = input_hash - stored_hash_obj

        if distance < smallest_distance:
            smallest_distance = distance
            best_match = card_entry

            # Afficher les détails de la comparaison pour le débogage
            # print(f"  Comparaison avec '{card_entry['name']}' (ID: {card_entry['id']}): Distance = {distance}")


    if best_match and smallest_distance <= max_hamming_distance:
        print(f"\nMeilleure correspondance trouvée : '{best_match['name']}' (ID: {best_match['id']})")
        print(f"Distance de Hamming : {smallest_distance}")
        return best_match
    else:
        print("\nAucune correspondance satisfaisante trouvée dans la base de données.")
        if best_match: # S'il y avait une meilleure correspondance mais qu'elle dépassait le seuil
            print(f"(La correspondance la plus proche était '{best_match['name']}' avec une distance de {smallest_distance}, ce qui est > {max_hamming_distance})")
        return None

if __name__ == "__main__":
    # Charger les données de hachage
    all_hashed_cards = load_hashed_data(HASHED_CARDS_JSON_PATH)

    if all_hashed_cards:
        
        input_image_file ="data_for_testing/charizard.jpg"


        # Pour pHash, des valeurs de 0 à 5 sont des correspondances très fortes, 5-10 sont de bonnes correspondances.
        HAMMING_DISTANCE_THRESHOLD = 14

        matched_card_info = find_matching_card(input_image_file, all_hashed_cards, max_hamming_distance=HAMMING_DISTANCE_THRESHOLD)

        # Si vous voulez tester avec une autre carte :
        # input_image_file_2 = "une_autre_carte.png"
        # find_matching_card(input_image_file_2, all_hashed_cards, max_hamming_distance=HAMMING_DISTANCE_THRESHOLD)