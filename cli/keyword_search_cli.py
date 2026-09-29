import argparse
import json, string, pickle, os, math
from nltk.stem import PorterStemmer
from collections import Counter

stemmer = PorterStemmer()

def sanitise_list(input: list[str]) -> list[str]:
    output = []
    for word in input:
        clean_word = preprocess_text(word)
        if clean_word not in stopwords:
            stemmed_word = stemmer.stem(clean_word)
            output.append(stemmed_word)
    return output

def remove_punctuation(input: str) -> str:
    punct = string.punctuation
    trans_table = str.maketrans(string.ascii_letters,string.ascii_letters, punct)
    output = input.translate(trans_table)
    return output

def preprocess_text(input: str) -> str:
    lowered = input.lower()
    no_punct = remove_punctuation(lowered)

    return no_punct

def tokenise(input: str) -> list[str]:
    sanitised = preprocess_text(input)
    output = sanitised.split()
    return sanitise_list(output)

def tokenise_word(term):
        output_list = tokenise(term)
        if len(output_list) > 1:
            raise Exception(f"Tokeniser outputted more than one word! Term: {term}")
        return output_list[0]

def check_partial_match(query: str) -> list[str]:
    output = []
    query_tokens = tokenise(query)
    cleaned_query_tokens = sanitise_list(query_tokens)
    index = InvertedIndex()
    index.load()
    for search_token in cleaned_query_tokens:
        found_ids = index.get_documents(search_token)
        for movie_id in found_ids:
            output.append(f"{movie_id} - {index.docmap[movie_id]['title']}")
            if len(output) >= 5:
                return output
        if len(output) >= 5:
            return output

    #print(f"Input elements: {query_tokens}")
    # for search_element in search_population:
    #     movie = search_element["title"]
    #     search_tokens = tokenise(movie)
    #     #cleaned_search_tokens = sanitise_list(search_tokens)
    #     #print(f"   Checking {search_element["title"]}: {search_tokens}")
    #     for search_token in search_tokens:
    #         for query_token in cleaned_query_tokens:
    #             if query_token in search_token:
    #                 #print(f"    Success - found {query_token} in {movie}")
    #                 output.append(movie)
    #                 break
    #         if movie in output:
    #             break

    return output

with open("data/stopwords.txt") as f:
    all_stopwords = f.read().splitlines()
    stopwords = [preprocess_text(x) for x in all_stopwords]

def load_movies(path:str) -> dict:
    with open(path) as f:
        movie_dict = json.load(f)
    return movie_dict

class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        tokenised_text = tokenise(text)
        for token in tokenised_text:
            #print(token)
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        return sorted(self.index.get(term, []))

    def build(self):
        movie_dict = load_movies("data/movies.json")["movies"]
        #print(movie_dict[0])
        for movie in movie_dict:
            doc_id = movie["id"]
            title = movie["title"]
            description = movie["description"]
            self.__add_document(doc_id, f"{title} {description}")
            self.docmap[doc_id] = movie
        
    def get_tf(self, doc_id, term):
        term = tokenise_word(term)
        if doc_id in self.term_frequencies.keys():
            doc_count = self.term_frequencies[doc_id]
            if term in doc_count:
                return doc_count[term]
        else:
            return 0


    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as index:
            pickle.dump(self.index, index)
        with open("cache/docmap.pkl", "wb") as docmap:
            pickle.dump(self.docmap, docmap)
        with open("cache/term_frequencies.pkl", "wb") as termfreq:
            pickle.dump(self.term_frequencies, termfreq)


    def load(self):
        try:
            with open("cache/index.pkl", "rb") as index:
                self.index = pickle.load(index)
            with open("cache/docmap.pkl", "rb") as docmap:
                self.docmap = pickle.load(docmap)
            with open("cache/term_frequencies.pkl", "rb") as termfreq:
                self.term_frequencies = pickle.load(termfreq)
        except Exception as e:
            print(f"Error: {e}")
            raise

    def get_idf(self, term) -> float:
        token = tokenise_word(term)
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(token, set()))
        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tf_idf(self, term, doc_id) -> float:
        token = tokenise_word(term)
        tf = self.get_tf(doc_id, token)
        idf = self.get_idf(token)
        tf_idf = tf * idf
        return tf_idf

    def get_bm25_idf(self, term:str) -> float:
        token = tokenise_word(term)
        total_doc_count = len(self.docmap)
        N = total_doc_count
        term_match_doc_count = len(self.index.get(token, set()))
        df = term_match_doc_count
        bm25 = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25
        
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    build_command = subparsers.add_parser("build", help="Builds an inverted index to assist in tokenised search")
    tf_command = subparsers.add_parser("tf", help="Tokenises the given term and looks up the term frequency")
    tf_command.add_argument("doc_id", type=int, help="Document ID to find term frequency for given token")
    tf_command.add_argument("term", type=str, help="Token")

    idf_command = subparsers.add_parser("idf", help="Returns an inverse document frequency value for a given term")
    idf_command.add_argument("term", type=str, help="Term to get IDF score for")

    bm25_idf_command = subparsers.add_parser("bm25idf", help="Returns an Okapi BM25 IDF value for a given term")
    bm25_idf_command.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    tfidf_command = subparsers.add_parser("tfidf", help="Returns TF-IDF score for given term and document ID")
    tfidf_command.add_argument("doc_id", type=int, help="Document ID to find TF-IDF for given token")
    tfidf_command.add_argument("term", type=str, help="Token")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            #movie_dict = load_movies("data/movies.json")
                #print(movie_dict)
            results = check_partial_match(args.query)
            # preprocessed_query = preprocess_text(args.query)
            # for movie in movie_dict["movies"]:
            #     preprocessed_movie = preprocess_text(movie["title"])
                
            #     if preprocessed_query in preprocessed_movie:
            #         results.append(movie["title"])
            print(results)
            for i in range(min(5, len(results))):
                print(f"{i+1}. {results[i]}")

        case "build":
            new_index = InvertedIndex()
            new_index.build()
            new_index.save()
            #docs = new_index.get_documents("merida")
            #print(f"First document for token 'merida' = {docs[0]}")

        case "tf":
            term = args.term
            term = tokenise_word(term)
            doc_id = args.doc_id
            index = InvertedIndex()
            index.load()
            term_frequency = index.get_tf(doc_id, term)
            print(term_frequency)

        case "idf":
            term = args.term
            index = InvertedIndex()
            index.load()
            idf = index.get_idf(term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")

        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            token = tokenise_word(term)
            index = InvertedIndex()
            index.load()
            tfidf = index.get_tf_idf(token, doc_id)
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")

        case "bm25idf":
            term = args.term
            index = InvertedIndex()
            index.load()
            bm25 = index.get_bm25_idf(term)
            print(f"BM25 IDF score of '{args.term}': {bm25:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()