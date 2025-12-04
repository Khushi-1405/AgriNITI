from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import api_view
from .models import User
from .serializers import UserSerializer

@api_view(['GET'])
def get_user(request):
    return Response(UserSerializer({'name':"Khushi", "age": 22}).data)

