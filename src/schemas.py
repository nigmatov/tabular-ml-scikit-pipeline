from pydantic import BaseModel

class InputRow(BaseModel):
    f1: float
    f2: float
    f3_cat: str
