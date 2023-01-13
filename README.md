# IR_project

Information Retrieval Final Project.
In our final project in information retrieval we build a search engine for English Wikipedia.

Description of the above file: 
1. serarch_frontend: python file which contains all search functions our serach engine performs. Full documentation of each function can be found in the file.
2. search_backend: python file which contains all background logic and functionality that helped us implement all search functions.
3. inverted_index_gcp: python file with implementation of Inverted Index.
4. TODO : notebook we used to build text invertd index
5. TODO : notebook we used to build title invertd index
6. TODO : notebook we used to build anchor invertd index

Files that we use in the the project but not in the repository:
1. DL.plk: pkl file contains a dictionary mapping document ID to his lenght.
2. doc_to_title.pkl: pkl file contains a dictionary mapping document ID to the document's title.
3. pageviews.pkl: pkl file contains a dictionary mapping document ID to his page view, which is a value represents the request for the content of a web page (relevant to the month of August 2021, more then 10.7 million viewed atricles).
4. pagerank: file contains a dictionary mapping document ID to his page rank value, which is a value represent the importance of a web page.
5. postings_gcp: folder contains index on the text of document (built using inverted_index_gcp).
6. postings_gcp_title: folder contains index on the title of document (built using inverted_index_gcp).
7. postings_gcp_anchor: folder contains index on the anchor of document (built using inverted_index_gcp).

