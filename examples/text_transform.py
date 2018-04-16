# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.embeddings


# Text to transform
text_to_transform = [u"", u"Hello, what is your name?", u"Hi! What time is it?", u"Hello, I am not there for the moment."]

# Show it
for text in text_to_transform:
    print(text)
# end for

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Token(),
    torchlanguage.transforms.ToIndex(start_ix=1),
    # torchlanguage.transforms.ToNGram(n=2, overlapse=True),
    torchlanguage.transforms.Embedding(torchlanguage.embeddings.CharacterEmbedding())
])

# Transformer
"""transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Character(),
    torchlanguage.transforms.ToIndex()
])"""

# Show it transformed
for text in text_to_transform:
    print(transformer(text))
# end for
