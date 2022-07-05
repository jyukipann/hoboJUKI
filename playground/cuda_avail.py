# python -c "import torch; print(torch.cuda.is_available());"

import torch

if torch.cuda.is_available():
    print("using cuda")
else:
    print("can't use cuda")