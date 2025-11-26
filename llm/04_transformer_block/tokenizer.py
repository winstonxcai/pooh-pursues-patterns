
def create_tokenizer(text: str) -> tuple[list, callable, callable]:
    """
    Create a tokenizer for the given text.

    Args:
        text: The text to create a tokenizer for.

    Returns:
        A list of all the tokens in the vocabulary, and two callable functions: encode and decode.
    """
    tokens = sorted(list(set(text)))
    
    stoi = {token: i for i, token in enumerate(tokens)}
    itos = {i: token for token, i in stoi.items()}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return tokens, encode, decode