# -*- coding: utf-8 -*-

# read it in to inspect it
#corpus = 'shakespeare.txt'
corpus = 'machado_de_assis_conto.txt'
corpus_name = corpus.split(".")[0]

with open('corpus/' + corpus, 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

def calculate_average_sentence_length(document, split='.'):
    sentences = document.split(split)  # Split the document into sentences based on the period (.)
    total_characters = 0
    total_sentences = 0

    for sentence in sentences:
        # Remove leading and trailing whitespace from each sentence
        sentence = sentence.strip()

        # Skip empty sentences
        if sentence:
            total_characters += len(sentence)
            total_sentences += 1

    if total_sentences == 0:
        return 0  # Return 0 if there are no sentences in the document
    else:
        return total_characters / total_sentences

# Example usage:
document = "Hello, how are you? I hope you're doing well. This is a sample document."
#average_length = calculate_average_sentence_length(document)
average_length = calculate_average_sentence_length(text, '\n')
print("Average sentence length in characters:", average_length)
