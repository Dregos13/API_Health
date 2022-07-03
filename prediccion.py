import pandas as pd
import database
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from bson.objectid import ObjectId

def predecir(identificador):

    print(identificador)

    client = database.function()
    db = client.usuarios

    archivo = db['archive']
    usuario = db['user']
    entrenamiento = db['entrenamiento']

    objInstance = ObjectId(identificador)


    df = pd.DataFrame(list(entrenamiento.find()))

    usuario = pd.DataFrame(list(usuario.find({"_id": objInstance})))

    archivo = pd.DataFrame(list(archivo.find()))

    df.drop('_id', inplace=True, axis=1)

    df['Cholesterol'] = df['Cholesterol'].astype(int)
    archivo['heartbeat'] = archivo['heartbeat'].astype(int)

    latido = archivo['heartbeat']

    usuario.drop('_id', inplace=True, axis=1)
    usuario.drop('email', inplace=True, axis=1)
    usuario.drop('roles', inplace=True, axis=1)
    usuario.drop('pass', inplace=True, axis=1)
    usuario.drop('name', inplace=True, axis=1)
    usuario.drop('alta', inplace=True, axis=1)
    usuario.drop('test', inplace=True, axis=1)
    usuario.drop('rol', inplace=True, axis=1)
    usuario.drop('_class', inplace=True, axis=1)
    df.drop('RestingECG', inplace=True, axis=1)
    df.drop('Oldpeak', inplace=True, axis=1)
    df.drop('ST_Slope', inplace=True, axis=1)

    usuario['MaxHR'] = latido.max()
    usuario =usuario[["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","MaxHR","ExerciseAngina"]]
    le = preprocessing.LabelEncoder()
    lclase = preprocessing.LabelEncoder()

    media = df.sum(axis = 0, skipna = True, numeric_only=int)
    resul  = media[4] / (len(df.index))
    df["Cholesterol"].replace({0 : resul}, inplace=True)

    df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
    df['Sex'] = le.fit_transform(df['Sex'])
    df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
    usuario['ChestPainType'] = le.fit_transform(usuario['ChestPainType'])
    usuario['Sex'] = le.fit_transform(usuario['Sex'])
    usuario['ExerciseAngina'] = le.fit_transform(usuario['ExerciseAngina'])

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    cols = [col for col in df.columns if col not in ['HeartDisease']]
    data = df[cols]
    target = df['HeartDisease']

    media = 0
    enfermedad = 0
    noEnfermedad = 0
    vpositivos = 0
    fpositivos = 0
    vnegativos = 0
    fnegativos = 0

    for train_index, test_index in kf.split(data):

        data_train2, data_test2 = data.iloc[train_index], data.iloc[test_index]

        target_train2, target_test2 = target.iloc[train_index], target.iloc[test_index]

        gnb = GaussianNB()

        model1 = gnb.fit(data_train2, target_train2)

        pred1 = model1.predict(data_test2)

        pred2 = model1.predict(usuario)

        if (pred2 == '1'):

            enfermedad = enfermedad + 1

        else:

            noEnfermedad = noEnfermedad + 1

        precision = accuracy_score(target_test2, pred1, normalize=True)

        m = confusion_matrix(target_test2, pred1)

        print(pred2)

        total = m.sum()

        vpositivos += m[0, 0]
        fpositivos += m[0, 1]
        vnegativos += m[1, 1]
        fnegativos += m[1, 0]

        media = media + precision

    media = media / 5

    return media, enfermedad, noEnfermedad