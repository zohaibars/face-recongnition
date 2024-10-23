from sqlalchemy import create_engine, Column, String, ForeignKey,INTEGER,UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, ForeignKey, LargeBinary
from sqlalchemy.orm import sessionmaker
import numpy as np
Base = declarative_base()

class Names(Base):
    __tablename__ = 'Names'
    id=Column(INTEGER,primary_key=True,autoincrement=True)
    name = Column(String,unique=True)

class Embeddings(Base):
    __tablename__ = 'Embeddings'
    id = Column(Integer, primary_key=True,autoincrement=True)  # Define id as the primary key
    name_id = Column(Integer, ForeignKey('Names.id'))  # Define foreign key
    embedding = Column(LargeBinary)

class FaceRecDB:
    def __init__(self):
        self.engine = create_engine('sqlite:///faceEmbeddings.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_name(self, name):
        session = self.Session()
        try:
            session.add(Names(name=name))
            session.commit()
            #print(f"Name '{name}' added successfully.")
        except Exception as e:
            session.rollback()
            #print(f"Error adding name '{name}': {e}")
        finally:
            session.close()
    def add_embedding(self, name, embedding):
        session = self.Session()
        try:
            id = self.get_id_by_name(name)
            if id is not None:
                # Convert NumPy array to bytes
                embedding_bytes = embedding.tobytes()
                session.add(Embeddings(name_id=id, embedding=embedding_bytes))
                session.commit()
                #print("Added embedding to database.")
        except Exception as e:
            session.rollback()
            #print(f"Error adding embedding for '{name}': {e}")
        finally:
            session.close()
    def add_name_embedding(self, data):
        for name, embedding in data.items():
            if self.get_id_by_name(name) is None:
                self.add_name(name)
                if isinstance(embedding, np.ndarray):
                    self.add_embedding(name, embedding)
                    #print(f"Added {name} & {len(embedding)} to database")
                else:
                    print(f"Embedding for {name} is not a NumPy array.")
            else:
                print(f"{name} already exists")

    def get_id_by_name(self, name):
        session = self.Session()
        try:
            name_id = session.query(Names.id).filter(Names.name == name).first()
            if name_id is not None:
                return name_id[0]
            else:
                #print(f"No ID found for name '{name}'.")
                return None
        except Exception as e:
            #print(f"Error retrieving ID for '{name}': {e}")
            return None
        finally:
            session.close()
    def get_embeddings_by_person_id(self, person_id):
        session = self.Session()
        try:
            # Query Embeddings table based on person_id
            embeddings = session.query(Embeddings).filter_by(name_id=person_id).all()
            
            # Process each embedding
            processed_embeddings = []
            for embedding in embeddings:
                processed_embedding = np.frombuffer(embedding.embedding, dtype=np.float32)  # Assuming dtype is float32
                processed_embeddings.append(processed_embedding)
            
            return processed_embeddings
                
        except Exception as e:
            #print(f"Error fetching embeddings: {e}")
            return None
        finally:
            session.close()
    def get_name_by_id(self,id):
        session = self.Session()
        try:
            name = session.query(Names.name).filter(Names.id == id).first()
            if name is not None:
                return name[0]
            else:
                #print(f"No name found for id '{id}'.")
                return None
        except Exception as e:
            #print(f"Error retrieving name for id '{id}': {e}")
            return None
        finally:
            session.close()
    def modify_name(self, original_name, new_name):
        session = self.Session()
        try:
            name_obj = session.query(Names).filter(Names.name == original_name).first()
            if name_obj:
                name_obj.name = new_name
                session.commit()
                #print(f"Name '{original_name}' in Names table modified to '{new_name}' successfully.")
            else:
                print(f"No entry found for name '{original_name}' in Names table.")
        except Exception as e:
            session.rollback()
            #print(f"Error modifying name '{original_name}' to '{new_name}' in Names table: {e}")
        finally:
            session.close()
        
    def search_embedding(self, embedding):
        session = self.Session()
        try:
            # Convert NumPy array to bytes
            embedding_bytes = embedding.tobytes()
            
            embedding_obj = session.query(Embeddings).filter(Embeddings.embedding == embedding_bytes).first()
            if embedding_obj:
                name = self.get_name_by_id(embedding_obj.id)
                if name:
                    return name
            return None
        except Exception as e:
            session.rollback()
            #print(f"Error searching for embedding: {e}")
            return None
        finally:
            session.close()

    def get_data(self):
        session = self.Session()
        # from sqlalchemy import outerjoin

        
        try:
            # Create aliases for Names and Embeddings tables
            query_result = session.query(Embeddings, Names).outerjoin(Names, Embeddings.name_id == Names.id).all()

            # Get the length of the result
            result_length = len(query_result)
            #print(result_length)
            # Process the query result
            names=[]
            embedding_list=[]
            for embedding, name in query_result:
                embedding_value=np.frombuffer(embedding.embedding, dtype=np.float32)
                name_value = name.name if name else None  # Handle the case where there's no corresponding name
                names.append(name_value)
                embedding_list.append(embedding_value)
                

            return np.array(names),np.array(embedding_list)
        except Exception as e:
            session.rollback()
            #print(f"Error fetching data: {e}")
            return None
        finally:
            session.close()

    def get_data_up(self):
        session=self.Session()
        try:
            names_query = session.query(Names.id, Names.name).all()
            names_dict = {id_: name for id_, name in names_query}

            # Query to get all embeddings with their corresponding IDs
            embeddings_query = session.query(Embeddings.id, Embeddings.embedding).all()
            embeddings_dict = {id_: embedding for id_, embedding in embeddings_query}

            combined_list = []
            for id_, name in names_dict.items():
                if id_ in embeddings_dict:
                    combined_list.append((name, embeddings_dict[id_]))
            return combined_list
        except Exception as e:
            return None, None
        finally:
            session.close()
# if __name__ == "__main__":
    # db = FaceRecDB()
    # npz_data = np.load(r'D:\MWaqar\FR\static\feature\face_features.npz')
    # data = npz_data['arr2'][0]
    # print(data.dtype)
    # print('Original array shape:', data.shape)
    # index=np.where(npz_data['arr1']=='Hamza_Shahbaz')[0]
    # orignal=npz_data['arr2'][index]
    # Convert array to bytes
    # bytes_data = data.tobytes()
    # print("data dtypes",data.dtype)
    # # Reconstruct array from bytes
    # reconstructed_array = np.frombuffer(bytes_data, dtype=data.dtype).reshape(data.shape)
    # print('Reconstructed array shape:', reconstructed_array.shape)

    # Check if the reconstructed array matches the original array
    # if np.array_equal(data, reconstructed_array):
    #     print('Arrays are equal')
    # else:
    #     print('Arrays are not equal')
    # for key in npz_data.keys():
    #     print(npz_data[key].shape)
    # Iterate over arrays in the npz file
    # add data to db *************************************************************
    # for i in range(len(npz_data['arr1'])):
    #     print(npz_data['arr1'][i])
    #     db.add_name(npz_data['arr1'][i])
    #     db.add_embedding(npz_data['arr1'][i],npz_data['arr2'][i])

    # get get data *8&**********************************************
    # a,b=db.get_data()
    # print(a)
    # print(b)
    # print(type(a))
    # print(type(b))
    # print(len(a))
    # print(len(b))
    #     # db.add_name_embedding({:})
    # # # print(npz_data['arr2'][0])
    # # data=npz_data['arr2'][0]
    # # print('conversion of bytes')
    # # bytes_data=data.tobytes()
    # # back=np.frombuffer(bytes_data)
    # # print("conversion completed",data.shape,back.shape)
    # names_array, embeddings_array=db.get_data()
    # print(names_array[:5])
    # print(names_array.shape,embeddings_array.shape)
    # print(orignal.shape)
    # id=db.get_id_by_name('Hamza_Shahbaz')
    # print(id)
    # embedding=db.get_embeddings_by_person_id(id)
    # embedding=np.array(embedding)
    # print(embedding.shape)
    
    # arraywise_equality = np.array_equal(orignal, embedding)
    # print(arraywise_equality)
        # db.add_embedding()
        # Convert array to binary representation
        # array_bytes = array.tobytes()
        