from django.urls import path
from .views import chatbot_response

# urlpatterns = [
#     path('chat/', chatbot_response, name='chatbot_response'),
# ]
from django.urls import path
from .views import chatbot_page, chatbot_response

urlpatterns = [
    path('', chatbot_page, name='chatbot'), 
    path('response/', chatbot_response, name='chatbot_response'),
]
