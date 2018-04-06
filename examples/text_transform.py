# -*- coding: utf-8 -*-
#

# Imports
import torchlanguage.transforms


# Text to transform
text_to_transform = [u"Hello, what is your name?", u"Hi! What time is it?", u"Hello, I am not there for the moment."]

# Show it
for text in text_to_transform:
    print(text)
# end for

# Transformer
transformer = torchlanguage.transforms.Compose([
    torchlanguage.transforms.Tag(),
    torchlanguage.transforms.ToIndex(),
    torchlanguage.transforms.ToOneHot(voc_size=15)
])

# Show it transformed
for text in text_to_transform:
    print(transformer(text))
# end for
