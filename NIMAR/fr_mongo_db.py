import numpy as np
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['FacialRecogniton_Nimar']  # Replace 'your_database_name' with your actual database name
collection = db['FR_Nimar'] 
def save_all_data():
    # Load data from xyz.npz file
    data = np.load('D:/MWaqar/FR/static/feature/face_features.npz')
    arr1 = data['arr1']
    arr2 = data['arr2']
    name_values_dict = {}

    # Loop through arr1 and arr2 to find and store corresponding values for each name
    for i, name in enumerate(arr1):
        if name not in name_values_dict:
            name_values_dict[name] = []
        name_values_dict[name].append(arr2[i].tolist())  # Convert numpy array to list

    # Insert values into MongoDB
    for name, values in name_values_dict.items():
        collection.insert_one({'name': name, 'values': values})

    # Close MongoDB connection
    

def fetch_all_data():
    
    all_documents = collection.find()

    # Initialize lists to store names and embeddings
    names_list = []
    embeddings_list = []

    # Iterate over the documents and extract names and embeddings
    for document in all_documents:
        names = document['name']
        embeddings = document['values']  # Assuming 'values' contains embeddings
        for embedding in embeddings:
            names_list.append(names)
            embeddings_list.append(embedding)

    # Convert lists to NumPy arrays
    names_array = np.array(names_list)
    embeddings_array = np.array(embeddings_list)
    
    return names_array,embeddings_array

def add_name_with_embedding(nam_embedding):
    name, embedding = next(iter(nam_embedding.items()))
    existing_entry = collection.find_one({'name': name})
    if existing_entry:
        print(f"Name '{name}' already exists in the database.")
    else:
        collection.insert_one({'name': name, 'values': embedding.tolist()})
        print(f"Name '{name}' added to the database.")

def add_embedding_to_existing_name(nam_embedding):
    name, embedding = next(iter(nam_embedding.items()))
    existing_entry = collection.find_one({'name': name})
    if existing_entry:
        existing_embeddings = existing_entry['values']
        existing_embeddings.append(embedding.tolist())
        collection.update_one({'name': name}, {'$set': {'values': existing_embeddings}})
        print(f"Additional embedding added for name '{name}'.")
    else:
        collection.insert_one({'name': name, 'values': [embedding.tolist()]})
        print(f"Name '{name}' added to the database with the new embedding.")

def add_name_with_embedding_training(name, embedding):
    existing_entry = collection.find_one({'name': name})
    print("existing", existing_entry)
    if existing_entry:
        print(f"Name '{name}' already exists in the database.")
    else:
        try:
            print(len(embedding))
            collection.insert_one({'name': name, 'values': embedding})  # Convert to list before inserting
        except Exception as e:
            print(f"Exception is {e}")
        print(f"Name '{name}' added to the database.")


def delete_all_unknown():
    filter_query = {'name': {'$regex': '^Unknown'}}

    # Delete records matching the filter query
    result = collection.delete_many(filter_query)

    # Print the number of deleted records
    print("Number of records deleted:", result.deleted_count)

    # Close MongoDB connection
def delete_person_by_name(name_to_delete):
    filter_query = {'name': name_to_delete}
    result = collection.delete_one(filter_query)

    # Check if a record was deleted
    if result.deleted_count == 1:
        print("Record with name '{}' has been deleted.".format(name_to_delete))
    else:
        print("No record with name '{}' found to delete.".format(name_to_delete))


def update_embedding_with_name(name, embedding):
    
    document = collection.find_one({'name': name})

    if document:
        # Extract existing embeddings
        existing_embeddings = document['values']
        # Append the new embedding
        existing_embeddings.append(embedding.tolist())  # Convert NumPy array to list before appending
        
        # Update the document with the new embeddings
        collection.update_one({'name': name}, {'$set': {'values': existing_embeddings}})
        print("Embedding appended successfully for name:", name)
    else:
        print("Name", name, "not found in the collection")
        print("Name", name, "not found in the collection")

    # Close MongoDB connection
    

def get_embedding_by_name(name):
    
    document = collection.find_one({'name': name})

    if document:
        # Extract embeddings from the document
        embeddings = document['values']
        print("Embeddings for", name, ":", embeddings)
        return np.array(embeddings)  # Convert list to NumPy array and return embeddings
    else:
        print("Name", name, "not found in the collection")

    # Close MongoDB connection
    
    
def update_name(existing_name,new_name):
    document = collection.find_one({'name': existing_name})

    if document:
        # Update the document with the new name
        collection.update_one({'name': existing_name}, {'$set': {'name': new_name}})
        print("Name replaced successfully: '{}' replaced with '{}'".format(existing_name, new_name))
    else:
        print("Name", existing_name, "not found in the collection")


def get_count():
    unknown_names = collection.find({'name': {'$regex': '^Unknown\d+$'}})
    
    # Extract numbers from the names and find the maximum
    max_number = 0
    for document in unknown_names:
        name = document['name']
        number = int(name.split('Unknown')[1])
        max_number = max(max_number, number)
    
    # Increment the maximum number by 1 to get the next number
    next_number = max_number + 1
    return next_number


save_all_data()
# data = np.load('face_features.npz')
# arr1 = data['arr1']
# arr2 = data['arr2']
# arr_list=[]
# embd=arr2[5]
# print(type(embd))
# for arr in arr2[:2]:
#     arr_list.append(arr)

# delete_person_by_name("zohaib")
# save_all_data()
# names_array,embeddings_array=fetch_all_data()
# print(names_array.shape,embeddings_array.shape)

# add_name_with_embedding(name="zohaib",embedding=np.array(arr_list))
# update_embedding_with_name(name="zohaib",embedding=embd)

# delete_all_unknown()
# names_array,embeddings_array=fetch_all_data()
# print(names_array.shape,embeddings_array.shape)
# embeddings=get_embedding_by_name("KAsif")
# print(embeddings.shape)

# update_name("KAsif","Khawaja_Asif")
# delete_person_by_name("Unknown52")

# import pymongo
# import numpy as np

# def save_npz_data_to_mongodb(npz_file_path, collection_name):
    # Connect to MongoDB
    # client = pymongo.MongoClient("mongodb://localhost:27017/")
    # db = client["FacialRecognitionEmbeddings"]
    # collection = db[collection_name]

    # Load data from NPZ file
    # npz_data = np.load(npz_file_path)
    # print(npz_data['arr1'])
    # print(npz_data['arr2'])
    # Create a list to hold the documents
    # documents = []

    # # Iterate through arrays in the NPZ file
    # for i in range(len(npz_data['arr1'])):
    #     # Convert NumPy array to nested Python list
    #     array_list = npz_data['arr2'][i].tolist()
    #     #print(npz_data['arr1'][i], len(array_list))

    #     # Create a document for each array
    #     document = {npz_data['arr1'][i]: array_list}
    #     documents.append(document)

    # # Insert all documents into MongoDB
    # if documents:
    #     collection.insert_many(documents)
    #     print("Data saved to MongoDB successfully!")
    # else:
    #     print("No data to save.")

# def retrieve_data_by_name(name, collection_name):
#     # Connect to MongoDB
#     client = pymongo.MongoClient("mongodb://localhost:27017/")
#     db = client["FacialRecognitionEmbeddings"]
#     collection = db[collection_name]

#     # Query the collection by name
#     result = collection.find_one({name: {"$exists": True}})
#     if result:
#         # Convert nested Python list back to NumPy array
#         retrieved_array = np.array(result[name])
#         return retrieved_array
#     else:
#         print("Data not found for the specified name.")
#         return None

# Example usage
# npz_file_path = "face_features.npz"  # Path to your NPZ file
# collection_name = "FR"  # Name of MongoDB collection

# save_npz_data_to_mongodb(npz_file_path, collection_name)

# # Retrieve data by name
# name_to_retrieve = "Hamza_Shahbaz"
# retrieved_data = retrieve_data_by_name(name_to_retrieve, collection_name)

# if retrieved_data is not None:
#     print(f"Retrieved data for {name_to_retrieve}:")
#     print(retrieved_data)
