# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms
import torchlanguage.embeddings


# Text to transform
text_to_transform = [u"Hello, what is your name?\nHi! What time is it?\nHello, I am not there for the moment.\nCount to 9!"]

# Show it
for text in text_to_transform:
    print(text)
# end for

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.RemoveLines(),
    torchlanguage.transforms.RemoveRegex(regex=r'(w|W)[a-z]+')
    # torchlanguage.transforms.Character(),
    # torchlanguage.transforms.ToIndex(),
    # torchlanguage.transforms.Embedding(torchlanguage.embeddings.CharacterEmbedding(n_gram=1, context=3, dim=10)),
    # torchlanguage.transforms.ToNGram(n=2)
])

# Show it transformed
for text in text_to_transform:
    print(transformer(text))
# end for
