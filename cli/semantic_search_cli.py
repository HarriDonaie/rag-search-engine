import argparse
from lib.semantic_search import * 


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Prints model information")

    embed_text_parser = subparsers.add_parser("embed_text", help="Gets an embedding from the model for the given text")
    embed_text_parser.add_argument("text", help="Text to be embedded")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Prints embeddings information")

    embed_query_parser = subparsers.add_parser("embed_query", help="Gets an embedding from the model for the given query text")
    embed_query_parser.add_argument("text", help="Query text to be embedded")

    search_parser = subparsers.add_parser("search", help="Searches a given query")
    search_parser.add_argument("query", type=str, help="Query text to be searched")
    search_parser.add_argument("--limit", nargs = "?", default = 5, type=int, help="Optional: Number of results to return. Default: 5")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.text)
        case "search":
            search = SemanticSearch()
            with open("data/movies.json") as f:
                movies_dict = json.load(f)
                documents = movies_dict["movies"]
            embeddings = search.load_or_create_embeddings(documents)
            result = search.search(args.query, args.limit)
            #print(result)
            for i in range(len(result)):
                title = result[i][1]["title"]
                score = result[i][0]
                description = result[i][1]["description"]
                print(f"{i+1}. {title} (score: {score:.4f})")
                print(description)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()