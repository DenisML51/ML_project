from pydantic import BaseModel 

class User(BaseModel):
    name: str 
    id: int 
    age: int
    is_adult: bool

class FeedBack(BaseModel):
    name: str 
    message: str 