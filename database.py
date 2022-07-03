from pymongo import MongoClient

def function():

    client = MongoClient("mongodb://localhost:27017")

    return client