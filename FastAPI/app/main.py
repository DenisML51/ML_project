# from fastapi import FastAPI
# import uvicorn
# from app.models.models import User, FeedBack
# from pydantic import BaseModel

# app = FastAPI()

# list = []

# @app.get('/user')
# async def user():
#     user = {
#         'name' : 'john',
#         'id': 123,
#         'age': 24,
#         'is_adult': False
#     }

#     user1: User = User(**user)

#     print(user1.name)

#     return user1

# @app.post('/user')
# async def adult(user: User):

#     if user.age >= 18:
#         user.is_adult = True
#     return user

# # Пример пользовательских данных (для демонстрационных целей) 
# fake_users = {
#     1: {"username": "john_doe", "email": "john@example.com"},
#     2: {"username": "jane_smith", "email": "jane@example.com"},
# }

# # Конечная точка для получения информации о пользователе по ID
# @app.get("/users/{user_id}")
# def read_user(user_id: int):
#     if user_id in fake_users:

#         print(fake_users[user_id])
#         return fake_users[user_id]
#     return {"error": "User not found"}

# class test(BaseModel):

#     input_2: str

# @app.get('/{test_input}')
# async def test_inp(test_input: str):
#     data = {

#         'input_2': test_input
#     }

#     data_fill: test = test(**data)
#     print(f'получен ввод: {data_fill.input_2}')
#     return data_fill

# @app.post('/feedback')
# async def send_feedback(feedback: FeedBack):
#     list.append({
#         'name':feedback.name,
#         'message': feedback.message
#     })

#     return f'Feedback recieved. Thank you {feedback.name}'