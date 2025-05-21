import json
import requests
from PIL import Image
import imagehash # Bibliothèque pour le hachage perceptuel
from io import BytesIO
import os

# Imports pour SQLAlchemy et la configuration de la base de données
from sqlalchemy import create_engine, Column, String, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Utiliser la variable d'environnement pour la connexion
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("Erreur : La variable d'environnement DATABASE_URL n'est pas définie.")
    print("Veuillez créer un fichier .env avec DATABASE_URL=postgresql://user:password@host:port/database")
    exit()

# Définition de la base SQLAlchemy
Base = declarative_base()

# --- Définitions des modèles SQLAlchemy (identiques à votre code) ---
class PokemonSet(Base):
    __tablename__ = 'pokemon_sets'
    id = Column(String, primary_key=True)
    name = Column(String)
    series = Column(String)
    release_date = Column(Date)
    logo_image = Column(String)

class PokemonCard(Base):
    __tablename__ = 'pokemon_card'
    id = Column(String, primary_key=True)
    name = Column(String)
    supertype = Column(String)
    rarity = Column(String)
    types = Column(String)
    artist = Column(String)
    image_url_large = Column(String)
    set_id = Column(String, ForeignKey('pokemon_sets.id'))

class Choice(Base): # Inclus pour la complétude, même si non utilisé directement ici
    __tablename__ = 'choice'
    choice_id = Column(String, primary_key=True)
    card1_id = Column(String)
    card2_id = Column(String)
    fav_card_id = Column(String)
# --- Fin des définitions des modèles SQLAlchemy ---

# Chemin vers le fichier JSON de sortie
HASHED_CARDS_JSON_PATH = "pokemon_card_hashes_from_db.json"

def download_image(url):
    """Télécharge une image depuis une URL et la retourne en tant qu'objet Image Pillow."""
    if not url:
        print("  URL de l'image non fournie, sautée.")
        return None
    try:
        response = requests.get(url, timeout=15) # Timeout un peu plus long pour les téléchargements
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.Timeout:
        print(f"  Timeout lors du téléchargement de {url}")
    except requests.exceptions.RequestException as e:
        print(f"  Erreur de téléchargement de {url}: {e}")
    except IOError as e:
        print(f"  Erreur lors de l'ouverture de l'image depuis {url}: {e}")
    return None

def hash_all_cards_from_db(db_engine, output_json_path):
    """
    Récupère toutes les cartes de la base de données, calcule leurs hachages perceptuels
    et les stocke dans un fichier JSON.
    """
    Session = sessionmaker(bind=db_engine)
    session = Session()

    hashed_cards_list = []

    try:
        print("Récupération des cartes depuis la base de données...")
        # Récupérer uniquement les colonnes nécessaires
        cards_from_db = session.query(PokemonCard.id, PokemonCard.name, PokemonCard.image_url_large).all()
        total_cards = len(cards_from_db)
        print(f"{total_cards} cartes trouvées dans la base de données.")

        if total_cards == 0:
            print("Aucune carte trouvée dans la base de données. Le fichier JSON sera vide.")
            # Écrire un JSON vide si aucune carte n'est trouvée
            with open(output_json_path, 'w') as f:
                json.dump([], f, indent=4)
            print(f"Fichier JSON vide créé à {output_json_path}")
            return


        for index, card_record in enumerate(cards_from_db):
            card_id, card_name, image_url = card_record
            print(f"\nTraitement de la carte {index + 1}/{total_cards}: {card_name} (ID: {card_id})")

            if not image_url:
                print(f"  Aucune URL d'image trouvée pour {card_name} (ID: {card_id}), sautée.")
                continue

            pil_image = download_image(image_url)

            if pil_image:
                try:
                    hash_value = imagehash.phash(pil_image)
                    hashed_cards_list.append({
                        "id": card_id,
                        "name": card_name,
                        "hash": str(hash_value)
                    })
                    print(f"  Hachage calculé : {hash_value}")
                except Exception as e:
                    print(f"  Erreur lors du hachage de l'image pour {card_name}: {e}")
            else:
                print(f"  Impossible de télécharger ou d'ouvrir l'image pour {card_name} (depuis {image_url}), sautée.")

        # Stocker la liste des hachages dans un fichier JSON
        with open(output_json_path, 'w') as f:
            json.dump(hashed_cards_list, f, indent=4)
        print(f"\nLes données de hachage des cartes ont été stockées dans {output_json_path}")

    except Exception as e:
        print(f"Une erreur générale s'est produite : {e}")
    finally:
        session.close()
        print("Session de base de données fermée.")


if __name__ == "__main__":
    print("Début du processus de hachage des cartes depuis la base de données PostgreSQL...")

    # Créer l'engine SQLAlchemy
    try:
        engine = create_engine(DATABASE_URL)
        # Optionnel : tester la connexion
        with engine.connect() as connection:
            print("Connexion à la base de données établie avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création de l'engine ou de la connexion à la base de données : {e}")
        print("Vérifiez votre chaîne de connexion DATABASE_URL et que le serveur PostgreSQL est accessible.")
        exit()

    # S'assurer que les tables existent (ne les crée pas si elles existent déjà)
    # Base.metadata.create_all(engine) # Décommentez si vous voulez créer les tables si elles n'existent pas.
                                     # Attention : cela ne met pas à jour les tables si elles existent déjà avec une structure différente.
                                     # Pour les migrations de schéma, utilisez des outils comme Alembic.

    hash_all_cards_from_db(engine, HASHED_CARDS_JSON_PATH)
    print("Processus de hachage terminé.")