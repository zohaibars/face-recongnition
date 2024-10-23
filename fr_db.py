import numpy as np
from db_fr import FaceRecDB
# Load the .npz file
data = np.load('face_features.npz')
print(data)
print(data['arr1'])
print(data['arr2'])
db=FaceRecDB()
# for i in range(len(data['arr1'])):
#     print(data['arr1'][i])
#     if not data['arr1'][i].startswith("Unknown"):
#         try:
#             db.add_name(data['arr1'][i])
#         except:
#             pass
#         db.add_embedding(data['arr1'][i],data['arr2'][i])
combined_list=db.get_data_up()
print(len(combined_list))
for data in combined_list:
    if data[0]=='Huma':
        print(True)
# images_names, images_embs=db.get_data()
# print(images_names.shape,images_embs.shape)
# print(np.count_nonzero(images_names=='Hamza_Shahbaz'),"ddddddddd")
# indices = np.where(images_names == 'Huma')
# print(indices)

# # Print the shape of corresponding elements in b
# for index in indices:
#     print(f"Shape of b[{index}]: {images_embs[index].shape}")
# print("images_names",images_names)
# Create a dictionary to store the mapping
# mapping = {key: data[key] for key in data.keys()}
# # # Print the mapping
# embedings={}
# for name, array in zip(mapping['arr1'], mapping['arr2']):
#     # print(f"{name}: {len(array)}")
#     embedings[name]=array
#     print(name)
#     print(array)
#     db.add_name(name)
#     db.add_embedding(name, str(array))
#     print("=" * 10)

# print(len(embedings))
# npz_data = np.load(r'face_features.npz')
# data = npz_data['arr2'][0]
# print(data.dtype)
# print('Original array shape:', data.shape)
# index=np.where(npz_data['arr1']=='Hamza_Shahbaz')[0]
# orignal=npz_data['arr2'][index]
# # Convert array to bytes
# bytes_data = data.tobytes()
# print("data dtypes",data.dtype)
# # Reconstruct array from bytes
# reconstructed_array = np.frombuffer(bytes_data, dtype=data.dtype).reshape(data.shape)
# print('Reconstructed array shape:', reconstructed_array.shape)