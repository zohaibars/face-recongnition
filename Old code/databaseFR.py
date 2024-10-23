from sqlalchemy import create_engine, Column, String, ForeignKey,INTEGER,UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
Base = declarative_base()

class Names(Base):
    __tablename__ = 'Names'
    id=Column(INTEGER,primary_key=True,autoincrement=True)
    name = Column(String,unique=True)

class Embeddings(Base):
    __tablename__ = 'Embeddings'
    id = Column(INTEGER, ForeignKey('Names.id'))
    embedding = Column(String,primary_key=True)

class FaceRecDB:
    def __init__(self):
        self.engine = create_engine('sqlite:///database.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        print("Database Intialized")

    def add_name(self, name):
        session = self.Session()
        try:
            session.add(Names(name=name))
            session.commit()
            print(f"Name '{name}' added successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error adding name '{name}': {e}")
        finally:
            session.close()
    def get_id_by_name(self, name):
        session = self.Session()
        try:
            name_id = session.query(Names.id).filter(Names.name == name).first()
            if name_id is not None:
                return name_id[0]
            else:
                print(f"No ID found for name '{name}'.")
                return None
        except Exception as e:
            print(f"Error retrieving ID for '{name}': {e}")
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
                print(f"No name found for id '{id}'.")
                return None
        except Exception as e:
            print(f"Error retrieving name for id '{id}': {e}")
            return None
        finally:
            session.close()
    def add_embedding(self, name, embedding):
        session = self.Session()
        try:
            id=self.get_id_by_name(name)
            if id !=None:
                session.add(Embeddings(id=id, embedding=embedding))
                session.commit()
                print("Added embedding to database.")
        except Exception as e:
            session.rollback()
            print(f"Error adding embedding for '{name}': {e}")
        finally:
            session.close()
    def add_name_embedding(self,data):
        for name, embedding in data.items():
            if self.get_id_by_name(name) is None:
                self.add_name(name)
                if not isinstance(embedding, str):
                    embedding = str(embedding)
                    self.add_embedding(name, embedding)
                    print(f"Added {name} & {len(embedding)} to database")
                else:
                    self.add_embedding(name, embedding)
                    print(f"Added {name} & {len(embedding)} to database")
            else:
                print(f"{name} already exsists")
    def modify_name(self, original_name, new_name):
        session = self.Session()
        try:
            name_obj = session.query(Names).filter(Names.name == original_name).first()
            if name_obj:
                name_obj.name = new_name
                session.commit()
                print(f"Name '{original_name}' in Names table modified to '{new_name}' successfully.")
            else:
                print(f"No entry found for name '{original_name}' in Names table.")
        except Exception as e:
            session.rollback()
            print(f"Error modifying name '{original_name}' to '{new_name}' in Names table: {e}")
        finally:
            session.close()
    def search_embedding(self,embedding:str)->str:
        session = self.Session()
        try:
            embedding = session.query(Embeddings).filter(Embeddings.embedding == embedding).first()
            if embedding:
                name=self.get_name_by_id(embedding.id)
                if name:
                    return name
                else:
                    return None
            else:
                return None
        except Exception as e:
            session.rollback()
            print(f"Error searching for embedding '{embedding}': {e}")
        finally:
            session.close()
        
    def get_data(self):
        session = self.Session()
        try:
            # Perform a join query to get data from both Names and Embeddings tables based on matching id
            join_query = session.query(Names.name, Embeddings.embedding).join(Embeddings, Names.id == Embeddings.id).all()
            print(join_query)
            # Extract names and embeddings into separate lists
            names = [result.name for result in join_query]
            # embeddings = [np.array(result.embedding) for result in join_query]
            embeddings=[]
            for result in join_query:
                emb=np.array(result.embedding)
                embeddings.append(emb)
            # Convert lists to numpy arrays
            names_array = np.array(names)
            embeddings_array = np.array(embeddings)
            # print(names_array)
            # print(len(names_array))
            # print(type(names_array))
            # # print(embeddings_array)
            # print(len(embeddings_array))
            # print(type(embeddings_array))
            return names_array, embeddings_array
        except Exception as e:
            session.rollback()
            print(f"Error fetching data: {e}")
            return None, None
        finally:
            session.close()


# if __name__ == "__main__":
#     db = FaceRecDB()
#     a,b=db.get_data()
#     # db.modify_name("Unknown1","Zain Asif")
#     name=db.search_embedding('''[-1.97048858e-02 -1.12346806e-01 -2.55014431e-02 -3.93441431e-02
#  -2.63477135e-02  2.65914332e-02 -6.38469383e-02  8.13216250e-03
#   1.54043408e-02  2.64075361e-02  2.75829434e-02 -4.66623120e-02
#   1.03521481e-04 -1.99084636e-02 -3.71561130e-03 -6.34036288e-02
#   5.98735623e-02 -7.20721623e-03 -2.20770147e-02 -1.51315704e-03
#   6.32005781e-02 -1.39106279e-02 -4.25868817e-02  1.78685989e-02
#  -5.46380319e-02 -1.02253156e-02 -3.31729054e-02  2.70718951e-02
#  -1.63870363e-03 -4.33747657e-02  7.78190419e-03 -1.38738181e-03
#   4.91104424e-02  4.15508151e-02  6.89291628e-03  8.48688127e-04
#   1.41784467e-03 -2.07497645e-02  9.93043333e-02  9.56514105e-02
#   3.67711894e-02  6.71178475e-02 -1.89826638e-02 -2.85394099e-02
#   6.17507286e-02 -5.10506630e-02  2.34537311e-02 -4.85835150e-02
#  -7.12413248e-03  1.99899804e-02 -6.10817457e-03  1.53090479e-02
#  -2.20397823e-02  1.65083344e-04  1.48154087e-02 -8.80852807e-03
#   1.28851843e-03  3.70331295e-02 -7.74347782e-02  2.01348420e-02
#   7.79501870e-02  3.61718773e-03  2.73027625e-02 -2.20121834e-02
#  -3.14911231e-02 -2.57599428e-02 -2.62429044e-02 -1.55434748e-02
#  -6.19454980e-02 -5.41212335e-02  7.69507512e-02 -6.07226640e-02
#  -2.22698003e-02  1.14366107e-01  4.58879843e-02 -1.81923639e-02
#  -8.94564728e-04  6.03787899e-02 -3.37179042e-02  3.68765444e-02
#  -2.20263023e-02  4.72254977e-02  1.80273708e-02 -7.84416031e-03
#  -2.15229131e-02  2.88719758e-02  9.88741685e-03  2.36894041e-02
#  -3.90463136e-02 -1.88229512e-02 -1.89872943e-02  1.03534036e-03
#   2.28929874e-02  1.09728426e-02 -9.77029558e-03  1.55041274e-02
#  -1.55634750e-02  7.47268274e-03  2.57182345e-02  8.87290835e-02
#  -4.01107371e-02 -1.10255824e-02 -5.59170954e-02 -3.74292433e-02
#   1.16370041e-02 -1.15230633e-03 -1.13173639e-02 -1.92442015e-02
#   3.42942774e-02  4.75684255e-02 -9.07694697e-02  7.66658187e-02
#   1.62281357e-02 -6.03270344e-02  6.40733391e-02 -2.96138767e-02
#  -7.31341541e-02  2.31539793e-02  9.01066512e-02 -5.46913296e-02
#  -3.42033617e-02 -4.17633392e-02  8.73625129e-02  3.68083976e-02
#   2.02243752e-03  2.21120361e-02 -1.27496114e-02  9.97937866e-04
#   5.10098301e-02 -7.29408339e-02  1.25920353e-02  8.19730610e-02
#  -4.68617044e-02  4.04265113e-02 -9.39572230e-02 -3.26454379e-02
#  -2.84102876e-02  3.38720307e-02 -6.80535333e-03 -3.45132761e-02
#  -6.85813054e-02  8.57085288e-02  7.42755160e-02 -6.03087321e-02
#   1.81883678e-01  2.70402757e-03  1.01550035e-02  4.02683243e-02
#  -8.13116953e-02 -1.42879803e-02  2.21715346e-02 -6.26313686e-02
#  -3.71585488e-02  6.97184121e-03 -3.31859500e-03  3.05976775e-02
#  -1.09163798e-01 -3.96965370e-02  9.47444700e-03 -7.73742050e-02
#  -2.24325415e-02  4.58029807e-02 -5.60455956e-03 -5.37856761e-03
#   1.33526744e-02  3.99548858e-02 -3.11195944e-02  2.80803405e-02
#   2.86763553e-02  5.36377020e-02  7.97783490e-03 -4.41991575e-02
#   6.25831708e-02 -8.01501144e-03  8.58247047e-04  3.17312367e-02
#  -6.79655671e-02  1.17681799e-02 -1.29989274e-02  1.33742252e-02
#  -5.92045020e-03  9.89193446e-04  6.07705042e-02  1.20557183e-02
#   6.43911362e-02  1.68903973e-02 -2.08043624e-02 -2.51265578e-02
#   3.91730433e-03 -5.19238822e-02  3.74539159e-02 -4.99171913e-02
#  -1.69895608e-02 -2.20793318e-02 -5.79738878e-02 -9.67454817e-03
#  -5.03088050e-02 -5.61438054e-02 -1.47406766e-02 -6.29810691e-02
#   5.77577055e-02 -4.30701226e-02  4.73579615e-02 -6.65408745e-03
#  -2.80325022e-02  6.79187402e-02 -9.06438455e-02 -5.25431111e-02
#  -9.05876085e-02  7.28471950e-02 -2.74848808e-02  5.02988249e-02
#  -3.19150500e-02  3.08051780e-02  9.70846117e-02  7.64421513e-03
#  -3.24939452e-02 -7.53562450e-02  2.63914317e-02  2.69680051e-03
#  -1.85498819e-02  2.83224862e-02 -2.92525310e-02  2.46486701e-02
#   5.08927852e-02  5.89795150e-02 -1.17572285e-02 -3.83415036e-02
#  -4.58335802e-02  2.35863104e-02 -4.04968299e-03  4.44409758e-04
#   3.03091686e-02  4.64196764e-02 -2.38824617e-02  1.81525778e-02
#   2.51842346e-02 -3.47268954e-02 -2.08709389e-03  4.05557156e-02
#   9.57298744e-03 -5.47503009e-02  5.54640498e-03  4.91204113e-02
#  -1.56756165e-03  3.62866186e-02 -6.38442561e-02  1.24077080e-02
#  -3.81481610e-02 -2.35598553e-02 -2.65763961e-02 -7.54627809e-02
#   8.43789894e-03 -9.39273834e-03  4.00191955e-02  3.47356759e-02
#   2.10725237e-02  7.30800852e-02 -4.59756106e-02 -5.10537578e-03
#  -7.29471212e-03 -1.26402695e-02 -2.93940026e-02  2.15856619e-02
#   1.13581689e-02 -4.01737019e-02  3.53845060e-02  2.27523651e-02
#   1.37976417e-02  4.37288322e-02  4.89498042e-02 -1.02533028e-02
#  -1.45142283e-02  2.92991474e-02 -2.55429223e-02  1.32822134e-02
#  -7.09033757e-02 -5.83432391e-02 -6.67160153e-02 -9.77567360e-02
#  -9.25858598e-03 -1.12318277e-01 -1.80277135e-03 -6.05196245e-02
#  -2.20083911e-02 -6.53364882e-02  7.51990303e-02  7.36099575e-03
#  -3.41441296e-02  3.12010124e-02 -3.25312689e-02 -1.80546939e-03
#  -8.07298627e-03 -2.80807670e-02  5.34821749e-02 -2.13858093e-05
#  -5.11473306e-02 -4.36800756e-02 -1.28624029e-04  3.83618549e-02
#  -9.88041237e-02 -1.87659673e-02 -3.03192921e-02  1.50246788e-02
#  -9.04318225e-03  3.91698517e-02 -9.62268785e-02 -9.11991373e-02
#   6.76654056e-02  6.64887158e-03  1.28148710e-02  2.55707558e-03
#  -2.60201981e-03 -1.15967561e-02  2.67173667e-02 -1.32412426e-02
#  -2.87444927e-02 -3.15282419e-02 -5.90172224e-02 -2.42616683e-02
#  -5.11511452e-02 -6.50418848e-02 -2.14639343e-02  9.39220889e-04
#   3.29880901e-02  2.24540550e-02  2.48084087e-02  5.18774278e-02
#  -2.67470814e-02  1.17989764e-01 -1.84284989e-02 -8.28052908e-02
#   5.09443246e-02  4.56171222e-02 -3.09491307e-02 -3.04306671e-02
#  -1.23453876e-02  7.47565180e-02 -3.00989766e-02 -1.33895278e-02
#   1.66035891e-02 -5.91731220e-02 -6.99573755e-02 -5.19286357e-02
#  -7.37734046e-03 -6.73953891e-02  3.83783579e-02  1.82142574e-02
#   6.05772994e-02  6.74800947e-02  2.48073768e-02 -4.64789495e-02
#   8.08598939e-03 -6.87597413e-03  4.24489267e-02 -7.28484103e-03
#  -6.30326048e-02  6.58014640e-02  2.68412824e-03  3.44273038e-02
#   1.15607055e-02 -1.87161621e-02 -6.41595796e-02  1.92524008e-02
#   1.51776234e-02  3.88803557e-02  6.18474744e-03  1.19472025e-02
#  -7.06750602e-02 -7.03603402e-02  4.63074036e-02 -3.83023135e-02
#  -1.01028522e-02  3.97718977e-03  3.16371955e-02 -6.28182068e-02
#   6.46106526e-02  6.66714162e-02  2.00675353e-02 -6.78403974e-02
#   3.37904282e-02 -5.29715531e-02 -5.98165058e-02 -4.72896919e-02
#   3.20588760e-02 -3.35177518e-02 -1.13064796e-02  1.57753993e-02
#   4.90119644e-02  5.06571047e-02  5.96613996e-02 -5.24156503e-02
#  -1.01157082e-02  7.21372990e-03  1.30169522e-02  7.08983764e-02
#   4.32824306e-02  3.64246853e-02 -4.11240384e-03 -2.97034197e-02
#  -5.70121035e-02 -1.10919615e-02  4.88702320e-02 -3.43485661e-02
#   7.16111204e-03  2.06131302e-02  3.12509574e-02  4.56591249e-02
#   1.71547364e-02  1.08437173e-01  5.00402749e-02 -2.41954494e-02
#  -8.91087204e-02  6.39681797e-03  1.30158039e-02  2.71217972e-02
#  -2.60922387e-02  4.55150614e-03 -2.51123346e-02  4.47190963e-02
#   9.61173698e-03 -3.67278755e-02 -3.82125266e-02  4.65675332e-02
#   1.91280246e-02 -5.37013821e-02  2.25448236e-02  2.36368962e-02
#  -9.52074155e-02 -5.31801432e-02  6.73481449e-02 -9.57209803e-03
#  -2.86480766e-02 -5.15806489e-03 -1.43070687e-02  5.65218106e-02
#  -2.17128694e-02 -1.73978042e-02 -1.32159265e-02  4.19738255e-02
#  -3.02015264e-02  5.71547193e-04 -6.56833872e-02 -9.02435258e-02
#  -1.73403695e-02  1.35801956e-02  5.77393658e-02  5.90615869e-02
#  -3.33944820e-02  4.77661099e-03 -8.17173049e-02  1.17105776e-02
#   5.49493656e-02  3.93659025e-02 -3.71792801e-02  5.67369200e-02
#   5.79950772e-02  3.40481587e-02 -4.98310737e-02  5.69715984e-02
#  -2.11918913e-02  1.24709569e-02 -5.31157292e-02  3.50920809e-03
#  -5.59065258e-03 -7.31020421e-02  7.21228868e-03  5.30526787e-02
#  -1.91989448e-02 -2.27418281e-02 -1.87760703e-02  9.85261053e-03
#  -1.72818955e-02 -1.44008314e-02 -2.30406187e-02  1.67444104e-03
#   6.67217374e-03 -1.94161367e-02  1.63627777e-03  3.17731723e-02
#  -2.43023541e-02  2.62664389e-02  7.29042385e-03 -6.73684152e-03
#   4.38103937e-02  1.04822852e-02  2.03364063e-02 -6.32298067e-02
#   3.61404978e-02  9.68024284e-02  2.77417339e-02 -5.13840988e-02
#  -3.17395739e-02 -7.09115043e-02 -4.89877425e-02 -7.52052218e-02
#  -3.78385410e-02 -1.14530809e-02  1.48562714e-02  1.08429044e-02
#   6.90366775e-02  5.41710593e-02 -3.39109497e-03 -1.53694317e-01
#   3.02690938e-02  2.08542813e-02 -2.78967153e-02  3.99965644e-02
#   6.14588931e-02  7.48558640e-02  1.52483834e-02  7.00190663e-03]''')
# print(name)