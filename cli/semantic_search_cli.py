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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()