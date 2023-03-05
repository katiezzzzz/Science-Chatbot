"""
This file generates a dictionary of wiki pages around scientific topics.
Upon running this file, the dictionary will be stored in /openfabric-test/code_files/wiki_data_full.pickle
This dictionary will be later used in model training.
"""


import wikipedia
import wikipediaapi
import pickle
import os

PATH = os.path.dirname(os.path.realpath(__file__))
PATH += "/"

def find_pages(search_topics):
    """
    input: list of search topics
    return: list of string (wiki page titles)
    """
    wiki_pages = []
    for topic in search_topics:
        results = wikipedia.search(topic, results=300)
        for result in results:
            if result not in wiki_pages:
                wiki_pages.append(result)
    return wiki_pages

def data_gen(wiki_pages, wiki_wiki):
    """
    input: list of wiki pages, wikipediaapi, list of strings to remove
    return: a dictionary containing the contents of all given pages
    """
    wiki_data = {}
    for page in wiki_pages:
        text = wiki_wiki.page(page).text
        text = text.replace("\n", ' ')
        text = text.replace("\"", "'")
    wiki_data[page] = text.lower()
    return wiki_data

def pickle_data(filename, data):
    """
    save data to a pickle file
    """
    filehandler = open(filename,"wb")
    pickle.dump(data,filehandler)

if __name__ == "__main__":
    wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    search_topics = ['science', 'chemisry', 'physics', 'biology', 'sciences', 'cosmology', 'medicine', 'geology']
    # generate data
    wiki_pages = find_pages(search_topics)
    wiki_data = data_gen(wiki_pages, wiki_wiki)
    # store data in pickle file
    pickle_data(PATH+"wiki_data_full.pickle", wiki_data)


