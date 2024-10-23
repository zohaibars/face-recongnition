
import os
import numpy as np
import logging
from pymongo import MongoClient
from app.utils.connections import get_collection, get_database
from app.utils.settings import COLLECTION_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = get_database()
collection = get_collection()

client = MongoClient('mongodb://localhost:27017/')

def save_all_features():
    current_directory = os.getcwd()
    path = os.path.join(current_directory, 'static/feature/face_features.npz')
    data = np.load(path)
    arr1 = data['arr1']
    arr2 = data['arr2']
    name_values_dict = {}

    for i, name in enumerate(arr1):
        if name not in name_values_dict:
            name_values_dict[name] = []
        name_values_dict[name].append(arr2[i].tolist())

    for name, values in name_values_dict.items():
        collection.insert_one({'name': name, 'values': values})

def load_to_mongo():
    logger.info("Saving all the data")
    current_directory = os.getcwd()
    path = os.path.join(current_directory, 'static/feature/face_features.npz')
    data = np.load(path)
    arr1 = data['arr1']
    arr2 = data['arr2']
    name_values_dict = {}

    for i, name in enumerate(arr1):
        if name not in name_values_dict:
            name_values_dict[name] = []
        name_values_dict[name].append(arr2[i].tolist())

    for name, values in name_values_dict.items():
        collection.insert_one({'name': name, 'values': values})
    logger.info("Loaded successfully")

def fetch_all_data():
    logger.info("Fetching all data")
    if COLLECTION_NAME not in db.list_collection_names():
        load_to_mongo()
        
    all_documents = collection.find()
    if not all_documents:
        load_to_mongo()
    names_list = []
    embeddings_list = []

    for document in all_documents:
        names = document['name']
        embeddings = document['values']
        for embedding in embeddings:
            names_list.append(names)
            embeddings_list.append(embedding)

    names_array = np.array(names_list)
    embeddings_array = np.array(embeddings_list)
    return names_array, embeddings_array

def add_name_with_embedding(nam_embedding):
    logger.info("Inserting embedings for new names.")
    name, embedding = next(iter(nam_embedding.items()))
    try:
        existing_entry = collection.find_one({'name': name})
    except Exception as ex:
        logger.error(f"Error while inserting{ex}")
    if existing_entry:
        logger.info(f"Name '{name}' already exists in the database.")
    else:
        collection.insert_one({'name': name, 'values': embedding.tolist()})
        logger.info(f"Name '{name}' added to the database.")

def add_embedding_to_existing_name(nam_embedding):
    logger.info("Inserting embedings for existing names.")
    name, embedding = next(iter(nam_embedding.items()))
    existing_entry = collection.find_one({'name': name})
    logger.info(existing_entry)
    if existing_entry:
        existing_embeddings = existing_entry['values']
        existing_embeddings.append(embedding.tolist())
        collection.update_one({'name': name}, {'$set': {'values': existing_embeddings}})
        logger.info(f"Additional embedding added for name '{name}'.")
    else:
        collection.insert_one({'name': name, 'values': [embedding.tolist()]})
        logger.info(f"Name '{name}' added to the database with the new embedding.")

def add_name_with_embedding_training(name, embedding):
    logger.info("Insert embedings with trained names")
    existing_entry = collection.find_one({'name': name})
    if existing_entry:
        logger.info(f"Name '{name}' already exists in the database.")
    else:
        try:
            collection.insert_one({'name': name, 'values': embedding})
        except Exception as e:
            logger.info(f"Exception is {e}")
        logger.info(f"Name '{name}' added to the database.")

def delete_all_unknown():
    logger.info("Deleting unknown faces")
    filter_query = {'name': {'$regex': '^Unknown'}}
    result = collection.delete_many(filter_query)
    logger.info("Number of records deleted:", result.deleted_count)
    return result.deleted_count

def delete_person_by_name(name_to_delete):
    logger.info("Deleting By name")

    filter_query = {'name': name_to_delete}
    result = collection.delete_one(filter_query)
    if result.deleted_count == 1:
        logger.info("Record with name '{}' has been deleted.".format(name_to_delete))
    else:
        logger.info("No record with name '{}' found to delete.".format(name_to_delete))

def update_embedding_with_name(name, embedding):
    logger.info("Updating by name")
    document = collection.find_one({'name': name})
    if document:
        existing_embeddings = document['values']
        existing_embeddings.append(embedding.tolist())
        collection.update_one({'name': name}, {'$set': {'values': existing_embeddings}})
        logger.info("Embedding appended successfully for name:", name)
    else:
        logger.info("Name", name, "not found in the collection")

def get_embedding_by_name(name):
    logger.info("Getting by name")
    document = collection.find_one({'name': name})
    if document:
        embeddings = document['values']
        logger.info("Embeddings for", name, ":", embeddings)
        return np.array(embeddings)
    else:
        logger.info("Name", name, "not found in the collection")

def update_name(existing_name, new_name):
    logger.info("Updating name")
    document = collection.find_one({'name': existing_name})
    if document:
        collection.update_one({'name': existing_name}, {'$set': {'name': new_name}})
        logger.info("Name replaced successfully: '{}' replaced with '{}'".format(existing_name, new_name))
    else:
        logger.info("Name", existing_name, "not found in the collection")

def get_count():
    logger.info("Getting count")
    unknown_names = collection.find({'name': {'$regex': '^Unknown\d+$'}})
    max_number = 0
    for document in unknown_names:
        name = document['name']
        number = int(name.split('Unknown')[1])
        max_number = max(max_number, number)
    next_number = max_number + 1
    return next_number

