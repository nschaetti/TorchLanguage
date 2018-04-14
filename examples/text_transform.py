# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.embeddings


# Text to transform
text_to_transform = [u"Hello, what is your name?", u"Hi! What time is it?", u"Hello, I am not there for the moment."]

# Show it
for text in text_to_transform:
    print(text)
# end for

# Transformer
"""transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.RemoveLines(),
    torchlanguage.transforms.RemoveRegex(regex=r'(w|W)[a-z]+')
])"""

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.GloveVector(),
    torchlanguage.transforms.DropOut(prob=0.1)
])

# Show it transformed
for text in text_to_transform:
    print(transformer(text))
# end for
