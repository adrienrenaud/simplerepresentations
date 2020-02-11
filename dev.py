from simplerepresentations import RepresentationModel


def load_data():
    return [
        "Le chat mange une pomme.",
        "Le chat mange une pomme.",
    ]


if __name__ == '__main__':
    model_type = 'flaubert'
    model_name = 'flaubert-base-uncased'

    representation_model = RepresentationModel(
        model_type=model_type,
        model_name=model_name,
        batch_size=6,
        max_seq_length=10, # truncate sentences to be less than or equal to 10 tokens
        combination_method='cat', # concatenate the last `last_hidden_to_use` hidden states
        last_hidden_to_use=4 # use the last 4 hidden states to build tokens representations
    )

    text_a = load_data()

    all_sentences_representations = representation_model(text_a=text_a)

    # print(all_sentences_representations[0]) # (2, 768) => (number of sentences, hidden size)
    # print(type(all_sentences_representations)) # (2, 768) => (number of sentences, hidden size)
    # print(all_sentences_representations.shape) # (2, 768) => (number of sentences, hidden size)
    # print(all_tokens_representations.shape) # (2, 10, 3072) => (number of sentences, number of tokens, hidden size)
