from typing import Any
import pytorch_lightning as pl


class Test(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        print(self.device)


model = Test()
print(model.device)